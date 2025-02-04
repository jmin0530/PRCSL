# -*- coding: utf-8 -*-
from enum import unique
import os
import time
import torch
import argparse
import importlib
import numpy as np
from functools import reduce
import torch.nn.functional as F
import torch.nn as nn
import math
from copy import deepcopy


import utils
import approach
from loggers.exp_logger import MultiLogger
from datasets.data_loader_split import get_final_datas, make_client_loaders
from datasets.dataset_config import dataset_config
from networks import tvmodels, allmodels, set_tvmodel_head_var
from networks.resnet18_fed import *
from networks.resnet32_fed import *
import datasets.memory_dataset as memd
import datasets.base_dataset as basedat
from networks.network import LLL_Net


# +
class Inc_Learning_Appr:
    """Basic class for implementing incremental learning approaches"""

    def __init__(self, server_model, client_model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False,
                 eval_on_train=False, exem_batch_size=128, logger: ExperimentLogger = None, exemplars_dataset: ExemplarsDataset = None):
        
        self.server_model = server_model
        self.client_model = client_model
        
        self.device = device
        self.nepochs = nepochs
        self.lr = lr
        self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad
        self.momentum = momentum
        self.wd = wd
        self.multi_softmax = multi_softmax
        self.logger = logger
        self.exemplars_dataset = exemplars_dataset
        self.warmup_epochs = wu_nepochs
        self.warmup_lr = lr * wu_lr_factor
        self.warmup_loss = torch.nn.CrossEntropyLoss()
        self.fix_bn = fix_bn
        self.eval_on_train = eval_on_train
        self.optimizer = None
        self.optimizer_server = None
        self.previous_client_loaders = None
        self.round_ = self.nepochs # 총 라운드 = 총 에폭
        self.local_epochs = 1 # 한 클라이언트의 local_epoch. 여기선 1로 설정
        self.exem_batch_size=exem_batch_size

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        return parser.parse_known_args(args)

    @staticmethod
    def exemplars_dataset_class():
        """Returns a exemplar dataset to use during the training if the approach needs it
        :return: ExemplarDataset class or None
        """
        return None

    def _get_optimizer(self):
        """Returns the optimizer"""
#         return torch.optim.SGD(self.server_model.parameters(), lr=self.lr, weight_decay=self.wd, momentum=self.momentum)
        return torch.optim.Adam(self.server_model.parameters(), lr=self.lr, weight_decay=self.wd)

    def train(self, t, client_loaders, taskcla):
        """Main train structure"""
#         self.pre_train_process(t, trn_loader)
        client_models = []
        for i in range(len(client_loaders)):
            client_models.append(self.client_model.to(self.device))
        self.optimizer_server = self._get_optimizer()
        self.train_loop(t, client_loaders, client_models)
        self.post_train_process(t, client_loaders)

#     # Federated averaging: FedAvg
#     def FedAvg(self, w, num_samples, total_num_clients):
#         # w: w_locals. client 갯수만큼 각 학습된 client가 존재
#         # n: 한 task에서 클라이언트들의 총 train data 갯수
#         n = 0
#         for c_n in num_samples:
#             n += c_n
#         w_avg = copy.deepcopy(w[0])
#         # n_k: 각 클라이언트의 train data 갯수
#         for k in w_avg.keys():
#             for i in range(total_num_clients):
#                 n_k = num_samples[i]
#                 if i == 0:
#                     w_avg[k] -= w[i][k]
#                 w_avg[k] += ((n_k)*w[i][k])
#             w_avg[k] = torch.div(w_avg[k], n)
#         return w_avg
    
#     # Federated averaging: FedAvg
#     def FedAvg(self, w, num_samples, total_num_clients):
#         w_avg = copy.deepcopy(w[0])
#         for k in w_avg.keys():
#             for i in range(1, len(w)):
#                 w_avg[k] += w[i][k]
#             w_avg[k] = torch.div(w_avg[k], len(w))
#         return w_avg
    
    def train_loop(self, t, client_loaders, client_models):
        """Contains the epochs loop"""
        
        lr = self.lr
        best_loss = np.inf
        patience = self.lr_patience
        best_server_model = self.server_model.get_copy()
        best_client_model = self.client_model.get_copy()
        num_clients = len(client_models)
        self.optimizer = self._get_optimizer()
        
        
        # Loop epochs
        for e in range(self.nepochs):
            clock0 = time.time()
            # t-1: self.server_model:   -> train_epoch.   -> t: self.server_model
            # Train one epoch
            self.train_epoch(t, client_loaders, client_models, lr)
            
            clock1 = time.time()
            print(' Epoch: {} | Train: time={:5.1f}s |'.format(e+1, clock1 - clock0), end='')
            
            
            # Validation
            clock3 = time.time()
            # 글로벌 모델로 val 하는게 정당한가?
            valid_loss, valid_acc, _ = self.eval(t, client_loaders, self.server_model, num_clients)
            clock4 = time.time()
            print(' Valid: time={:5.1f}s loss={:.3f}, TAw acc={:5.1f}% |'.format(
                clock4 - clock3, valid_loss, 100 * valid_acc), end='')
            self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=valid_loss, group="valid")
            self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * valid_acc, group="valid")

            # Update best validation loss
            if valid_loss < best_loss:
                best_loss = valid_loss
                patience = self.lr_patience
                best_server_model = self.server_model.get_copy()
                print(' *', end='')
            # If the loss does not go down, decrease patience
            else:
                patience -= 1
                if patience <= 0:
                    print()
                    break       

            self.logger.log_scalar(task=t, iter=e + 1, name="patience", value=patience, group="train")
            self.logger.log_scalar(task=t, iter=e + 1, name="lr", value=lr, group="train")
            print()
            
        self.server_model.set_state_dict(best_server_model)  # 최종 모델임 ;;

    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""
        pass

    def train_epoch(self, t, client_loaders, client_models, lr):
        """Runs a single epoch"""
        current_client_num = len(client_models)
        
        # 모델 담을라구.
        clients = []
        global_model_weights = copy.deepcopy(self.server_model.state_dict())
        
        # 각 클라이언트마다 자기 데이터로 학습함.
        for i in range(current_client_num):
            # Load a client
            client = client_models[i]
            client.train()

            
            # Receive previous client's weight
                
            if self.opt == 'adam':
                client_optim = torch.optim.Adam(client.parameters(), lr=lr, weight_decay=self.wd)
            elif self.opt == 'sgd':
                client_optim = torch.optim.SGD(client.parameters(), lr=lr, weight_decay=self.wd, momentum=self.momentum)

            loader = client_loaders[i][0]
        
            for images, targets in loader: 
                # Forward current client
                outputs = client(images.to(self.device))
                loss = self.criterion(t, outputs, targets.to(self.device), client)
                
                # Proximal term 추가
                prox_term = 0
                for param, global_param in zip(client.parameters(), global_model_weights.values()):
                    prox_term += ((param - global_param) ** 2).sum()
                    
                # Client backpropagation
                client_optim.zero_grad()
                loss += (mu / 2) * prox_term
                loss.backward()
                client_optim.step()
            # 로더 다 학습했으면 보냄.    
            clients.append(client)
        
        # fedavg 
        
        global_dict = self.server_model.state_dict()
        for key in global_dict.keys():
            global_dict[key] = torch.stack([clients[i].state_dict()[key] for i in range(current_client_num)], 0).mean(0)
        
        # 글로벌 서버모델 최신화
        self.server_model.load_state_dict(global_dict)
        
        
        
    def final_eval(self, t, client_loaders, num_clients, test=False, eval_eeil=False):
        """Contains the evaluation code"""
        total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
        for i in range(num_clients):
            if eval_eeil:
                loader = client_loaders[1][i][1]
            else:
                if test:
                    loader = client_loaders[i][2]
                else:
                    loader = client_loaders[i][1]
                
            with torch.no_grad():
                self.server_model.eval()
                for images, targets in loader:
                    # Forward current model
                    outputs = self.server_model(images.to(self.device))
                    loss = self.criterion(t, outputs, targets.to(self.device), self.server_model)
                    hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                    total_loss += loss.item() * len(targets)
                    total_acc_taw += hits_taw.sum().item()
                    total_acc_tag += hits_tag.sum().item()
                    total_num += len(targets)
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num
    
    
    def eval(self, t, client_val_loader, completed_client):
        """Contains the evaluation code"""
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
                    # Forward client model
                    outputs = completed_client(images.to(self.device))
                    
                    # Calculate loss
                    loss = self.criterion(t, outputs, y)

                    # Forward server model and evaluate
                    hits_taw, hits_tag = self.calculate_metrics(outputs, y)
                    
                    total_loss += loss.item()*len(targets)
                    total_acc_taw += hits_taw.sum().item()
                    total_acc_tag += hits_tag.sum().item()
                    total_num += len(targets)
        
        final_valid_loss = total_loss / total_num
        final_valid_acc_aw = total_acc_taw / total_num
        final_valid_acc_ag = total_acc_tag / total_num
        
        return final_valid_loss, final_valid_acc_aw, final_valid_acc_ag

    def calculate_metrics(self, outputs, targets):
        """Contains the main Task-Aware and Task-Agnostic metrics"""
        pred = torch.zeros_like(targets.to(self.device))
        # Task-Aware Multi-Head
        for m in range(len(pred)):
            this_task = (self.server_model.task_cls.cumsum(0) <= targets[m]).sum()
            pred[m] = outputs[this_task][m].argmax() + self.server_model.task_offset[this_task]
        hits_taw = (pred == targets.to(self.device)).float()
        # Task-Agnostic Multi-Head
        if self.multi_softmax:
            outputs = [torch.nn.functional.log_softmax(output, dim=1) for output in outputs]
            pred = torch.cat(outputs, dim=1).argmax(1)
        else:
            pred = torch.cat(outputs, dim=1).argmax(1)
        hits_tag = (pred == targets.to(self.device)).float()
        return hits_taw, hits_tag

    def criterion(self, t, outputs, targets, model):
        """Returns the loss value"""
        return torch.nn.functional.cross_entropy(outputs[t], targets - self.server_model.task_offset[t])
