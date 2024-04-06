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

    def _train_unbalanced(self, t, client_loaders, client_models):
        """Unbalanced training"""
        self._finetuning_balanced = False
        self._train_epoch = 0
        final_client_loaders = self._get_train_loader(client_loaders, False)
#         self.eeil_train_loop(t, final_client_loaders, client_models)
        super().train_loop(t, final_client_loaders, client_models)

    def _train_balanced(self, t, client_loaders, client_models):
        """Balanced finetuning"""
        self._finetuning_balanced = True
        self._train_epoch = 0
        orig_lr = self.lr
        self.lr *= self.lr_finetuning_factor
        orig_nepochs = self.nepochs
        self.nepochs = self.nepochs_finetuning
        final_client_loaders = self._get_train_loader(client_loaders, True)
#         self.eeil_train_loop(t, final_client_loaders, client_models)
        super().train_loop(t, final_client_loaders, client_models)
        self.lr = orig_lr
        self.nepochs = orig_nepochs

    def _get_train_loader(self, client_loaders, balanced=False):
        """Modify loader to be balanced or unbalanced"""
        new_client_dataloader = deepcopy(client_loaders)
        for i in range(len(client_loaders)):
            client_train_dataset = client_loaders[i][0].dataset
            if balanced:
                client_indices = torch.randperm(len(client_train_dataset))
                client_train_dataset = torch.utils.data.Subset(client_train_dataset, client_indices[:(len(self.exemplars_dataset)//len(client_loaders))])
            client_dataloader = DataLoader(client_train_dataset, batch_size=client_loaders[i][0].batch_size,
                                  shuffle=True,
                                  num_workers=client_loaders[i][0].num_workers,
                                  pin_memory=client_loaders[i][0].pin_memory)
            
            new_client_dataloader[i][0] = client_dataloader
        return new_client_dataloader

    def _noise_grad(self, parameters, iteration, eta=0.3, gamma=0.55):
        """Add noise to the gradients"""
        parameters = list(filter(lambda p: p.grad is not None, parameters))
        variance = eta / ((1 + iteration) ** gamma)
        for p in parameters:
            p.grad.add_(torch.randn(p.grad.shape, device=p.grad.device) * variance)

    def train_loop(self, t, client_loaders, client_models):
        """Contains the epochs loop"""
        if t == 0:  # First task is simple training
            super().train_loop(t, client_loaders, client_models)
        else:
            # exemplar dataset을 클라이언트 갯수만큼 나누어서 쪼개기
            self.exemplar_loaders_split = []
            n_clients = len(client_loaders)
            n_exemplars = len(self.exemplars_dataset.images)
            shuffled_exemplars_dataset = deepcopy(self.exemplars_dataset)
            exem_indices = torch.randperm(len(shuffled_exemplars_dataset.images))
            for i in range(n_clients):
                one_exem_data = deepcopy(self.exemplars_dataset)
                one_exem_data.images = list(np.array(shuffled_exemplars_dataset.images)\
                                            [exem_indices[i*(n_exemplars//n_clients):(i+1)*(n_exemplars//n_clients)]])
                one_exem_data.labels = list(np.array(shuffled_exemplars_dataset.labels)\
                                            [exem_indices[i*(n_exemplars//n_clients):(i+1)*(n_exemplars//n_clients)]])
                self.exemplar_loaders_split.append(torch.utils.data.DataLoader(one_exem_data,
                                                         batch_size=self.exem_batch_size,
                                                         shuffle=True, drop_last=False))
            
            print("Unbalanced training")
            self._train_unbalanced(t, client_loaders, client_models)
            
            # Balanced fine-tunning (new + old)
            print("Balanced training")
            self._train_balanced(t, client_loaders, client_models)

        # After task training： update exemplars
        self.exemplars_dataset.collect_exemplars(t, self.server_model, self.client_model, self.exemplars_dataset, \
                                                 client_loaders, client_loaders[0][1].dataset.transform, \
                                                 dp=True, prev_cls=self.prev_classes)
        
        # compute differentially private mean on a per-class basis
        if t == 0:
            class_nums = self.server_model.task_cls[0]
            self.first_exemplar_size = len(deepcopy(self.exemplars_dataset))
        else:
            class_nums = sum(self.server_model.task_cls[:t+1])
        
        for cur_cls in range(class_nums):
            cls_ind = np.where(np.array(list(self.exemplars_dataset.labels))==cur_cls)[0]
            if cur_cls in self.prev_classes:
                if self.exem_per_class == 0:
                    prev_cls_ind_first = cls_ind[self.first_exemplar_size//(self.dp_mean_batch*class_nums)]
                    prev_cls_ind_last = cls_ind[-1]
                    del self.exemplars_dataset.images[prev_cls_ind_first:prev_cls_ind_last+1]
                    del self.exemplars_dataset.labels[prev_cls_ind_first:prev_cls_ind_last+1]
                continue
            class_data = deepcopy(self.exemplars_dataset.images[cls_ind[0]:cls_ind[-1]+1])
            class_data_label = deepcopy([cur_cls for _ in range(len(cls_ind))])
                    
            class_data = list(torch.split(torch.tensor(class_data).detach(), self.dp_mean_batch))
            del self.exemplars_dataset.images[cls_ind[0]:cls_ind[-1]+1]
            del self.exemplars_dataset.labels[cls_ind[0]:cls_ind[-1]+1]
            for ind, (d, l) in enumerate(zip(class_data, class_data_label)):
                max_ = torch.max(d).item()
                min_ = torch.min(d).item()
                mean_image = torch.mean(d.type(torch.float32), dim=0).detach()
                if len(mean_image.shape) == 3:
                    h,w,c = mean_image.shape
                    dp_image = (mean_image + torch.tensor(np.random.laplace(loc=0, scale=max_/(self.dp_mean_batch*self.epsilon), \
                                                                                      size=(h,w,c)), dtype=torch.float32).detach()).numpy()
                else:
                    h,w = mean_image.shape
                    dp_image = (mean_image + torch.tensor(np.random.laplace(loc=0, scale=max_/(self.dp_mean_batch*self.epsilon), \
                                                                                      size=(h,w)), dtype=torch.float32).detach()).numpy()
                del mean_image
                dp_image = torch.clamp(torch.from_numpy(dp_image), min=0, max=255).numpy().astype(np.uint8)
                self.exemplars_dataset.images.append(dp_image)
                self.exemplars_dataset.labels.append(l)
                del dp_image
                
        print("Final memory size: ", len(self.exemplars_dataset.labels))
        
        # save previous classes
        self.prev_classes=deepcopy(np.unique(self.exemplars_dataset.labels))
        print("Previous classes: ", self.prev_classes)
           
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
        
        exemplar_dataloader = None
        if t > 0:
            exemplar_dataloader = torch.utils.data.DataLoader(self.exemplars_dataset,
                                                         batch_size=self.exem_batch_size,
                                                         shuffle=False, drop_last=False)
            current_client_num = len(client_models)+len(self.exemplar_loaders_split)
        else:
            current_client_num = len(client_models)
        
        k=0
        j=0
        for i in range(current_client_num):
            # 모델 client 초기화
            client = client_models[0]
            
            # 모델 client 이전 client것 불러오기
            if i > 0:
                client.load_state_dict(previous_client.state_dict())
            client.train()
            if t > 0:
                if i % 2 == 0:
                    loader = client_loaders[j][0]
                    now_exem=False
                    j += 1
                elif i % 2 == 1:
                    loader = self.exemplar_loaders_split[k]
                    now_exem=True
                    k += 1
                
            else:
                now_exem=False
                loader = client_loaders[j][0]
                j += 1
                
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
