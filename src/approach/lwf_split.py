# -*- coding: utf-8 -*-
import torch
from copy import deepcopy
from argparse import ArgumentParser
from copy import deepcopy

from .incremental_learning_split import Inc_Learning_Appr
from datasets.exemplars_dataset_split import ExemplarsDataset


# +
class Appr(Inc_Learning_Appr):
    """Class implementing the Learning Without Forgetting (LwF) approach
    described in https://arxiv.org/abs/1606.09282
    """

    # Weight decay of 0.0005 is used in the original article (page 4).
    # Page 4: "The warm-up step greatly enhances fine-tuning’s old-task performance, but is not so crucial to either our
    #  method or the compared Less Forgetting Learning (see Table 2(b))."
    def __init__(self, global_server_net, global_client_net, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False, eval_on_train=False, exem_batch_size=128,
                 logger=None, exemplars_dataset=None, lamb=1, T=2):
        super(Appr, self).__init__(global_server_net, global_client_net, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, exem_batch_size, logger,
                                   exemplars_dataset)
        self.server_model_old = None
        self.client_model_old = None
#         self.lamb = lamb
        self.T = T

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        # Page 5: "We use T=2 according to a grid search on a held out set, which aligns with the authors’
        #  recommendations." -- Using a higher value for T produces a softer probability distribution over classes.
        parser.add_argument('--T', default=2, type=int, required=False,
                            help='Temperature scaling (default=%(default)s)')
        return parser.parse_known_args(args)

    def _get_server_optimizer(self):
        """Returns the optimizer"""
        if len(self.exemplars_dataset) == 0 and len(self.server_model.heads) > 1:
            # if there are no exemplars, previous heads are not modified
            params = list(self.server_model.model.parameters()) + list(self.server_model.heads[-1].parameters())
        else:
            params = self.server_model.parameters()
    
        if self.opt == 'adam':
            return torch.optim.Adam(params, lr=self.lr, weight_decay=self.wd)
        elif self.opt == 'sgd':
            return torch.optim.SGD(params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum)


    def train_loop(self, t, client_loaders, client_models):
        """Contains the epochs loop"""

        # FINETUNING TRAINING -- contains the epochs loop
        super().train_loop(t, client_loaders, client_models)

        # EXEMPLAR MANAGEMENT -- select training subset
        
#         self.exemplars_dataset.collect_exemplars(t, self.server_model, self.client_model, \
#                                                  self.server_model.task_offset, client_loaders, \
#                                                  client_loaders[0][1].dataset.transform, dp=False, prev_cls=None, fix_prev=False)

    def post_train_process(self, t, client_loader):
        """Runs after training all the epochs of the task (after the train session)"""

        # Restore best and save model for future tasks
        self.server_model_old = deepcopy(self.server_model)
        self.server_model_old.eval()
        self.server_model_old.freeze_all()

        self.client_model_old = deepcopy(self.client_model)
        self.client_model_old.eval()
        self.client_model_old.freeze_all()

    def train_epoch(self, t, client_loaders, client_models, lr):
        """Runs a single epoch"""
        self.server_model.train()
        exemplar_dataloader = None
        current_client_num = len(client_models)
        
        j=0
        for i in range(current_client_num):
            # 모델 client 초기화
            client = client_models[0]
            
            # 모델 client 이전 client것 불러오기
            if i > 0:
                client.load_state_dict(previous_client.state_dict())
            client.train()
            
            if self.opt == 'adam':
                client_optim = torch.optim.Adam(client.parameters(), lr=lr, weight_decay=self.wd)
            elif self.opt == 'sgd':
                client_optim = torch.optim.SGD(client.parameters(), lr=lr, weight_decay=self.wd, momentum=self.momentum)

            loader = client_loaders[i][0]
                
            for images, targets in loader:
                outputs_old = None
                if t > 0:
                    client_old_outputs = self.client_model_old(images.to(self.device))
                    outputs_old = self.server_model_old(client_old_outputs)

                # Forward current model
                client_outputs = client(images.to(self.device))
                client_fx = client_outputs.clone().detach().requires_grad_(True)
                outputs = self.server_model(client_fx)
                loss = self.criterion(t, outputs, targets.to(self.device), outputs_old)

                # Server Backward
                self.optimizer_server.zero_grad()
                loss.backward()
                dfx_client = client_fx.grad.clone().detach()
                torch.nn.utils.clip_grad_norm_(self.server_model.parameters(), self.clipgrad)
                self.optimizer_server.step()

                # Client Backward
                client_optim.zero_grad()
                client_outputs.backward(dfx_client)
                torch.nn.utils.clip_grad_norm_(client.parameters(), self.clipgrad)
                client_optim.step()
                
            if i == current_client_num-1:
                return client
            else:
                previous_client = deepcopy(client).to(self.device)
                
    def evaluate_server(self, t, fx_client, y, len_batch, targets_old):
        self.server_model.to(self.device)
        self.server_model.eval()

        with torch.no_grad():
            fx_client = fx_client.to(self.device)
            y = y.to(self.device) 
            #---------forward prop-------------
            fx_server = self.server_model(fx_client)

            # calculate loss
            loss = self.criterion(t, fx_server, y, targets_old)
            
            # calculate accuracy
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            hits_taw, hits_tag = self.calculate_metrics(fx_server, y)
            
            # Log
            total_loss += loss.item() * len(y)
            total_acc_taw += hits_taw.sum().item()
            total_acc_tag += hits_tag.sum().item()
            total_num += len(y)
                
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    def eval(self, t, client_loaders, completed_client, num_clients, test=False):
        """Contains the evaluation code"""
        total_valid_loss, total_valid_acc_aw, total_valid_acc_ag = 0.0, 0.0, 0.0
        iteration = 0
        for i in range(num_clients):
            if test:
                loader = client_loaders[i][2]
            else:
                loader = client_loaders[i][1]
            with torch.no_grad():
                completed_client.eval()
                for images, targets in loader:
                    # Forward old model
                    targets_old = None
                    if t > 0:
                        client_old_outputs = self.client_model_old(images.to(self.device))
                        targets_old = self.server_model_old(client_old_outputs)
                        
                    # Forward current model
                    client_outputs = completed_client(images.to(self.device))
                    valid_loss, valid_acc_aw, valid_acc_ag = self.evaluate_server(t, client_outputs, targets.to(self.device), images.shape[0], targets_old)
                    total_valid_loss += valid_loss
                    total_valid_acc_aw += valid_acc_aw
                    total_valid_acc_ag += valid_acc_ag
                    iteration += 1
                    
        final_valid_loss = total_valid_loss / iteration
        final_valid_acc_aw = total_valid_acc_aw / iteration
        final_valid_acc_ag = total_valid_acc_ag / iteration                    
                    
        return final_valid_loss, final_valid_acc_aw, final_valid_acc_ag

    def cross_entropy(self, outputs, targets, exp=1.0, size_average=True, eps=1e-5):
        """Calculates cross-entropy with temperature scaling"""
        out = torch.nn.functional.softmax(outputs, dim=1)
        tar = torch.nn.functional.softmax(targets, dim=1)
        if exp != 1:
            out = out.pow(exp)
            out = out / out.sum(1).view(-1, 1).expand_as(out)
            tar = tar.pow(exp)
            tar = tar / tar.sum(1).view(-1, 1).expand_as(tar)
        out = out + eps / out.size(1)
        out = out / out.sum(1).view(-1, 1).expand_as(out)
        ce = -(tar * out.log()).sum(1)
        if size_average:
            ce = ce.mean()
        return ce

    def criterion(self, t, outputs, targets, outputs_old=None):
        """Returns the loss value"""
        loss = 0
        if t > 0:
            # Knowledge distillation loss for all previous tasks
            loss += self.lamb * self.cross_entropy(torch.cat(outputs[:t], dim=1),
                                                   torch.cat(outputs_old[:t], dim=1), exp=1.0 / self.T)
        # Current cross-entropy loss -- with exemplars use all heads
        if len(self.exemplars_dataset) > 0:
            return loss + torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
        return loss + torch.nn.functional.cross_entropy(outputs[t], targets - self.server_model.task_offset[t])
