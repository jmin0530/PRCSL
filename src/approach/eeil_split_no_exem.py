# -*- coding: utf-8 -*-
import torch
import warnings
from copy import deepcopy
from argparse import ArgumentParser
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np
import time

from .incremental_learning_split import Inc_Learning_Appr
from datasets.exemplars_dataset_split import ExemplarsDataset

import cv2


class Appr(Inc_Learning_Appr):
    """Class implementing the End-to-end Incremental Learning (EEIL) approach described in
    http://openaccess.thecvf.com/content_ECCV_2018/papers/Francisco_M._Castro_End-to-End_Incremental_Learning_ECCV_2018_paper.pdf
    Original code available at https://github.com/fmcp/EndToEndIncrementalLearning
    Helpful code from https://github.com/arthurdouillard/incremental_learning.pytorch
    """

    def __init__(self, server_model, client_model, device, nepochs=90, lr=0.1, lr_min=1e-6, lr_factor=10, lr_patience=5, clipgrad=10000,
                 momentum=0.9, wd=0.0001, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False,
                 eval_on_train=False, exem_batch_size=128, logger=None, exemplars_dataset=None, lamb=1.0, T=2, lr_finetuning_factor=0.1,
                 nepochs_finetuning=40, noise_grad=False):
        super(Appr, self).__init__(server_model, client_model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, exem_batch_size, logger,
                                   exemplars_dataset)
        self.model_old = None
        self.lamb = lamb
        self.T = T
        self.lr_finetuning_factor = lr_finetuning_factor
        self.nepochs_finetuning = nepochs_finetuning
        self.noise_grad = noise_grad

        self._train_epoch = 0
        self._finetuning_balanced = None
        
        self.prev_classes = []

        # EEIL is expected to be used with exemplars. If needed to be used without exemplars, overwrite here the
        # `_get_optimizer` function with the one in LwF and update the criterion
        have_exemplars = self.exemplars_dataset.max_num_exemplars + self.exemplars_dataset.max_num_exemplars_per_class
        if not have_exemplars:
            warnings.warn("Warning: EEIL is expected to use exemplars. Check documentation.")

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        # Page 6: "Based on our empirical results, we set T to 2 for all our experiments"
        parser.add_argument('--T', default=2.0, type=float, required=False,
                            help='Temperature scaling (default=%(default)s)')
        # "The same reduction is used in the case of fine-tuning, except that the starting rate is 0.01."
        parser.add_argument('--lr-finetuning-factor', default=0.1, type=float, required=False,
                            help='Finetuning learning rate factor (default=%(default)s)')
        # Number of epochs for balanced training
        parser.add_argument('--nepochs-finetuning', default=40, type=int, required=False,
                            help='Number of epochs for balanced training (default=%(default)s)')
        # the addition of noise to the gradients
        parser.add_argument('--noise-grad', action='store_true',
                            help='Add noise to gradients (default=%(default)s)')
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

    def _noise_grad(self, parameters, iteration, eta=0.3, gamma=0.55):
        """Add noise to the gradients"""
        parameters = list(filter(lambda p: p.grad is not None, parameters))
        variance = eta / ((1 + iteration) ** gamma)
        for p in parameters:
            p.grad.add_(torch.randn(p.grad.shape, device=p.grad.device) * variance)

    def train_loop(self, t, client_loaders, client_models):
        """Contains the epochs loop"""
        super().train_loop(t, client_loaders, client_models)
           
    def post_train_process(self, t, client_loaders):
        """Runs after training all the epochs of the task (after the train session)"""

        # Save old model to extract features later
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
                del images
                
                client_fx = client_outputs.clone().detach().requires_grad_(True)
                outputs = self.server_model(client_fx)
                loss = self.criterion(t, outputs, targets.to(self.device), outputs_old)
                del outputs
                if t > 0:
                    del outputs_old

                # Server Backward
                self.optimizer_server.zero_grad()
                loss.backward()
                dfx_client = client_fx.grad.clone().detach()
                torch.nn.utils.clip_grad_norm_(self.server_model.parameters(), self.clipgrad)
                self.optimizer_server.step()
                del loss
                del client_fx
                
                # Client Backward
                client_optim.zero_grad()
                client_outputs.backward(dfx_client)
                torch.nn.utils.clip_grad_norm_(client.parameters(), self.clipgrad)
                client_optim.step()
                del dfx_client
                
            if i == current_client_num-1:
                return client
            else:
                previous_client = deepcopy(client).to(self.device)

    def criterion(self, t, outputs, targets, outputs_old=None):
        # Classification loss for new classes
        loss = torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
        # Distilation loss
        if t > 0 and outputs_old:
            # take into account current head when doing balanced finetuning
            last_head_idx = t if self._finetuning_balanced else (t - 1)
            for i in range(last_head_idx):
                loss += self.lamb * F.binary_cross_entropy(F.softmax(outputs[i] / self.T, dim=1),
                                                           F.softmax(outputs_old[i] / self.T, dim=1))
        return loss
