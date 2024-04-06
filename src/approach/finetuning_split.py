# -*- coding: utf-8 -*-
import torch
from argparse import ArgumentParser
from copy import deepcopy

from .incremental_learning_split import Inc_Learning_Appr
from datasets.exemplars_dataset_split import ExemplarsDataset


class Appr(Inc_Learning_Appr):
    """Class implementing the finetuning baseline"""

    def __init__(self, global_server_model, global_client_model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False, eval_on_train=False, exem_batch_size=128,
                 logger=None, exemplars_dataset=None, all_outputs=False):
        super(Appr, self).__init__(global_server_model, global_client_model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, exem_batch_size, logger,
                                   exemplars_dataset)
        self.all_out = all_outputs

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--all-outputs', action='store_true', required=False,
                            help='Allow all weights related to all outputs to be modified (default=%(default)s)')
        return parser.parse_known_args(args)

    def _get_server_optimizer(self):
        """Returns the optimizer"""
        if len(self.exemplars_dataset) == 0 and len(self.server_model.heads) > 1 and not self.all_out:
            # if there are no exemplars, previous heads are not modified
            params = list(self.server_model.model.parameters()) + list(self.server_model.heads[-1].parameters()) # parameter 키 확인 필요
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
#         self.server_model...
#          self.client_model...
#         self.exemplars_dataset.collect_exemplars(t, self.server_model, self.client_model, \
#                                                  self.server_model.task_offset, client_loaders, \
#                                                  client_loaders[0][1].dataset.transform)
    
    def train_epoch(self, t, client_loaders, client_models, lr):
        """Runs a single epoch"""
        self.server_model.train()
        
        exemplar_dataloader = None
        current_client_num = len(client_models)
        
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
                # Forward current model
                client_outputs = client(images.to(self.device))
                client_fx = client_outputs.clone().detach().requires_grad_(True)
                outputs = self.server_model(client_fx)
                loss = self.criterion(t, outputs, targets.to(self.device))

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

    def criterion(self, t, outputs, targets):
        """Returns the loss value"""
        if self.all_out or len(self.exemplars_dataset) > 0:
            return torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
        return torch.nn.functional.cross_entropy(outputs[t], targets - self.server_model.task_offset[t])
