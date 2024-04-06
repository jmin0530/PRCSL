# -*- coding: utf-8 -*-
import torch
import itertools
from argparse import ArgumentParser
from copy import deepcopy

from .incremental_learning_split import Inc_Learning_Appr
from datasets.exemplars_dataset_split import ExemplarsDataset
from datasets.exemplars_selection_split import override_dataset_transform


class Appr(Inc_Learning_Appr):
    """Class implementing the Memory Aware Synapses (MAS) approach (global version)
    described in https://arxiv.org/abs/1711.09601
    Original code available at https://github.com/rahafaljundi/MAS-Memory-Aware-Synapses
    """

    def __init__(self, global_server_net, global_client_net, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False, eval_on_train=False, exem_batch_size=128,
                 logger=None, exemplars_dataset=None, lamb=1, alpha=0.5, fi_num_samples=-1):
        super(Appr, self).__init__(global_server_net, global_client_net, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train,exem_batch_size, logger,
                                   exemplars_dataset)
        self.lamb = lamb
        self.alpha = alpha
        self.num_samples = fi_num_samples

        # In all cases, we only keep importance weights for the model, but not for the heads.
        feat_ext_client = self.client_model
        feat_ext_server = self.server_model.model
        
        # Store current parameters as the initial parameters before first task starts
        self.older_params_client = {n: p.clone().detach() for n, p in feat_ext_client.named_parameters() if p.requires_grad}
        self.older_params_server = {n: p.clone().detach() for n, p in feat_ext_server.named_parameters() if p.requires_grad}
        
        # Store fisher information weight importance
        self.importance_client = {n: torch.zeros(p.shape).to(self.device) for n, p in feat_ext_client.named_parameters()
                           if p.requires_grad}
        self.importance_server = {n: torch.zeros(p.shape).to(self.device) for n, p in feat_ext_server.named_parameters()
                           if p.requires_grad}
        
    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        # Define how old and new importance is fused, by default it is a 50-50 fusion
        parser.add_argument('--alpha', default=0.5, type=float, required=False,
                            help='MAS alpha (default=%(default)s)')
        # Number of samples from train for estimating importance
        parser.add_argument('--fi-num-samples', default=-1, type=int, required=False,
                            help='Number of samples for Fisher information (-1: all available) (default=%(default)s)')
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

    # Section 4.1: MAS (global) is implemented since the paper shows is more efficient than l-MAS (local)
    def estimate_parameter_importance(self, t, client_loaders):
        # Initialize importance matrices
        importance_client = {n: torch.zeros(p.shape).to(self.device) for n, p in self.client_model.named_parameters()
                           if p.requires_grad}
        importance_server = {n: torch.zeros(p.shape).to(self.device) for n, p in self.server_model.model.named_parameters()
                           if p.requires_grad}
        
        if self.opt == 'adam':
            client_optimizer = torch.optim.Adam(self.client_model.parameters(), lr=self.optimizer_server.param_groups[0]['lr'], \
                                            weight_decay=self.wd)
        elif self.opt == 'sgd':
            client_optimizer = torch.optim.SGD(self.client_model.parameters(), lr=self.optimizer_server.param_groups[0]['lr'], \
                                            weight_decay=self.wd, momentum=self.momentum)
        
        # Do forward and backward pass to accumulate L2-loss gradients
        self.client_model.train()
        self.server_model.train()

        client_num = len(client_loaders)
        
        j=0
        n_samples = 0
        for i in range(client_num):
            # Compute fisher information for specified number of samples -- rounded to the batch size
            loader = client_loaders[i][0]
            if len(client_loaders[i][0].dataset) < client_loaders[i][0].batch_size:
                n_samples_batches = (len(client_loaders[i][0].dataset) // client_loaders[i][0].batch_size)+1
            else:
                n_samples_batches = (len(client_loaders[i][0].dataset) // client_loaders[i][0].batch_size)
                
                
            for images, targets in itertools.islice(loader, n_samples_batches):
                # MAS allows any unlabeled data to do the estimation, we choose the current data as in main experiments
                client_outputs = self.client_model.forward(images.to(self.device))
                fx_client = client_outputs.clone().detach().requires_grad_(True)
                outputs = self.server_model.forward(fx_client)
                
                # Page 6: labels not required, "...use the gradients of the squared L2-norm of the learned function output."
                loss = torch.norm(torch.cat(outputs, dim=1), p=2, dim=1).mean()
                
                client_optimizer.zero_grad()
                self.optimizer_server.zero_grad()
                loss.backward()
                dfx_client = fx_client.grad.clone().detach()
                client_outputs.backward(dfx_client)
                
                # Eq. 2: accumulate the gradients over the inputs to obtain importance weights
                for n, p in self.client_model.named_parameters():
                    if p.grad is not None:
                        importance_client[n] += p.grad.abs() * len(targets)
                        
                for n, p in self.server_model.model.named_parameters():
                    if p.grad is not None:
                        importance_server[n] += p.grad.abs() * len(targets)
                        
            n_samples += (n_samples_batches * loader.batch_size)
            
        # Eq. 2: divide by N total number of samples
        importance_client = {n: (p / n_samples) for n, p in importance_client.items()}
        importance_server = {n: (p / n_samples) for n, p in importance_server.items()}
            
        return importance_client, importance_server

    def train_loop(self, t, client_loaders, client_models):
        """Contains the epochs loop"""

        # FINETUNING TRAINING -- contains the epochs loop
        super().train_loop(t, client_loaders, client_models)

        # EXEMPLAR MANAGEMENT -- select training subset
#         self.exemplars_dataset.collect_exemplars(t, self.server_model, self.client_model, \
#                                                  self.server_model.task_offset, client_loaders, \
#                                                  client_loaders[0][1].dataset.transform)
        
    def train_epoch(self, t, client_loaders, client_models, lr, eeil=False):
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

    def post_train_process(self, t, client_loaders):
        """Runs after training all the epochs of the task (after the train session)"""

        # Store current parameters for the next task
        self.older_params_server = {n: p.clone().detach() for n, p in self.server_model.model.named_parameters() if p.requires_grad}
        self.older_params_client = {n: p.clone().detach() for n, p in self.client_model.named_parameters() if p.requires_grad}

        # calculate Fisher information
        curr_importance_client, curr_importance_server = self.estimate_parameter_importance(t, client_loaders)
        
        # merge fisher information, we do not want to keep fisher information for each task in memory
        for n in self.importance_client.keys():
            # Added option to accumulate importance over time with a pre-fixed growing alpha
            if self.alpha == -1:
                alpha = (sum(self.server_model.task_cls[:t]) / sum(self.server_model.task_cls)).to(self.device)
                self.importance_client[n] = alpha * self.importance_client[n] + (1 - alpha) * curr_importance_client[n]
            else:
                # As in original code: MAS_utils/MAS_based_Training.py line 638 -- just add prev and new
                self.importance_client[n] = self.alpha * self.importance_client[n] + (1 - self.alpha) * curr_importance_client[n]
        for n in self.importance_server.keys():
            # Added option to accumulate importance over time with a pre-fixed growing alpha
            if self.alpha == -1:
                alpha = (sum(self.server_model.task_cls[:t]) / sum(self.server_model.task_cls)).to(self.device)
                self.importance_server[n] = alpha * self.importance_server[n] + (1 - alpha) * curr_importance_server[n]
            else:
                # As in original code: MAS_utils/MAS_based_Training.py line 638 -- just add prev and new
                self.importance_server[n] = self.alpha * self.importance_server[n] + (1 - self.alpha) * curr_importance_server[n]

    def criterion(self, t, outputs, targets):
        """Returns the loss value"""
        loss = 0
        if t > 0:
            loss_reg = 0
            # Eq. 3: memory aware synapses regularizer penalty
            for n, p in self.client_model.named_parameters():
                if n in self.importance_client.keys():
                    loss_reg += torch.sum(self.importance_client[n] * (p - self.older_params_client[n]).pow(2)) / 2
                    
            for n, p in self.server_model.model.named_parameters():
                if n in self.importance_server.keys():
                    loss_reg += torch.sum(self.importance_server[n] * (p - self.older_params_server[n]).pow(2)) / 2
                    
            loss += self.mas_lamb * loss_reg
        # Current cross-entropy loss -- with exemplars use all heads
        if len(self.exemplars_dataset) > 0:
            return loss + torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
        return loss + torch.nn.functional.cross_entropy(outputs[t], targets - self.server_model.task_offset[t])
