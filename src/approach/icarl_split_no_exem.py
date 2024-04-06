# -*- coding: utf-8 -*-
import torch
import warnings
import numpy as np
from copy import deepcopy
from argparse import ArgumentParser
from torch.utils.data import DataLoader
import random

from .incremental_learning_split import Inc_Learning_Appr
from datasets.exemplars_dataset_split import ExemplarsDataset
from datasets.exemplars_selection_split import override_dataset_transform


class Appr(Inc_Learning_Appr):
    """Class implementing the Incremental Classifier and Representation Learning (iCaRL) approach
    described in https://arxiv.org/abs/1611.07725
    Original code available at https://github.com/srebuffi/iCaRL
    """

    def __init__(self, server_model, client_model, device, nepochs=60, lr=0.5, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0.9, wd=1e-5, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False,
                 eval_on_train=False, exem_batch_size=128, logger=None, exemplars_dataset=None, lamb=1):
        super(Appr, self).__init__(server_model, client_model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, exem_batch_size, logger,
                                   exemplars_dataset)
        self.server_model_old = None
        self.client_model_old = None
        self.total_exem_info = []
        self.previous_client_loaders = []
        self.prev_classes = []
        # iCaRL is expected to be used with exemplars. If needed to be used without exemplars, overwrite here the
        # `_get_optimizer` function with the one in LwF and update the criterion
        have_exemplars = self.exemplars_dataset.max_num_exemplars + self.exemplars_dataset.max_num_exemplars_per_class
        if not have_exemplars:
            warnings.warn("Warning: iCaRL is expected to use exemplars. Check documentation.")

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        # Sec. 4. " allowing iCaRL to balance between CE and distillation loss."
        parser.add_argument('--lamb', default=1, type=float, required=False,
                            help='Forgetting-intransigence trade-off (default=%(default)s)')
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

    # Algorithm 2: iCaRL Incremental Train
    def train_loop(self, t, client_loaders, client_models):
        """Contains the epochs loop"""
        super().train_loop(t, client_loaders, client_models)

        self.exemplars_dataset.collect_exemplars(t, self.server_model, self.client_model, \
                                                 self.server_model.task_offset, client_loaders, \
                                                 client_loaders[0][1].dataset.transform, \
                                                dp=True, prev_cls=self.prev_classes)
        
        
    def post_train_process(self, t, client_loaders):
        """Runs after training all the epochs of the task (after the train session)"""

        # Save old model to extract features later. This is different from the original approach, since they propose to
        #  extract the features and store them for future usage. However, when using data augmentation, it is easier to
        #  keep the model frozen and extract the features when needed.

            
        self.server_model_old = deepcopy(self.server_model)
        self.server_model_old.eval()
        self.server_model_old.freeze_all()

        self.client_model_old = deepcopy(self.client_model)
        self.client_model_old.eval()
        self.client_model_old.freeze_all()

    def train_epoch(self, t, client_loaders, client_models, lr):
        """Runs a single epoch"""
        self.server_model.train()
        current_client_num = len(client_models)

        for i in range(current_client_num):
            # 모델 client 초기화
            client = client_models[0]
            
            # 모델 client 이전 client것 불러오기
            if i > 0:
                client.load_state_dict(previous_client.state_dict())
            client.train()
            loader = client_loaders[i][0]
                
            if self.opt == 'adam':
                client_optim = torch.optim.Adam(client.parameters(), lr=lr, weight_decay=self.wd)
            elif self.opt == 'sgd':
                client_optim = torch.optim.SGD(client.parameters(), lr=lr, weight_decay=self.wd, momentum=self.momentum)
            else:
                raise Exception("Can't generate client optimizer!!")
                
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

    # Server-side functions associated with Testing
    def evaluate_server(self, t, fx_client, y, old_output):
        self.server_model.to(self.device)
        self.server_model.eval()

        with torch.no_grad():
            fx_client = fx_client.to(self.device)
            y = y.to(self.device) 
            fx_server = self.server_model(fx_client)

            # calculate loss
            loss = self.criterion(t, fx_server, y, old_output)
            
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
        total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
        iteration = 0
        for i in range(num_clients):
            if test:
                loader = client_loaders[i][2]
            else:
                loader = client_loaders[i][1]
            with torch.no_grad():
                completed_client.eval()
                for images, targets in loader:
#                     if targets.shape[0] != 1:
                    # Forward old model
                    outputs_old = None
                    if t > 0:
                        client_old_outputs = self.client_model_old(images.to(self.device))
                        outputs_old = self.server_model_old(client_old_outputs)

                    # Forward current model
                    client_outputs = completed_client(images.to(self.device))
                    valid_loss, valid_acc_aw, valid_acc_ag = self.evaluate_server(t, client_outputs, targets.to(self.device), \
                                                                                  outputs_old)
                    total_valid_loss += valid_loss
                    total_valid_acc_aw += valid_acc_aw
                    total_valid_acc_ag += valid_acc_ag
                    iteration += 1

        final_valid_loss = total_valid_loss / iteration
        final_valid_acc_aw = total_valid_acc_aw / iteration
        final_valid_acc_ag = total_valid_acc_ag / iteration
        return final_valid_loss, final_valid_acc_aw, final_valid_acc_ag


    # Algorithm 3: classification and distillation terms -- original formulation has no trade-off parameter (lamb=1)
    def criterion(self, t, outputs, targets, outputs_old=None):
        """Returns the loss value"""
        # Classification loss for new classes
        loss = torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
        
        # Distillation loss for old classes
        if t > 0:
            # The original code does not match with the paper equation, maybe sigmoid could be removed from g
            g = torch.sigmoid(torch.cat(outputs[:t], dim=1))
            q_i = torch.sigmoid(torch.cat(outputs_old[:t], dim=1))
            loss += self.lamb * sum(torch.nn.functional.binary_cross_entropy(g[:, y], q_i[:, y]) for y in
                                    range(sum(self.server_model.task_cls[:t])))
        return loss







