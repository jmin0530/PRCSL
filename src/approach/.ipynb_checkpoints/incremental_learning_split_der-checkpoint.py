# -*- coding: utf-8 -*-
import time
import torch
import numpy as np
from argparse import ArgumentParser
import copy

from loggers.exp_logger import ExperimentLogger
from datasets.exemplars_dataset_split import ExemplarsDataset

from .incremental_learning_split import Inc_Learning_Appr


class Inc_Learning_Appr(Inc_Learning_Appr):
    """Basic class for implementing incremental learning approaches"""

    def __init__(self, global_server_model, global_client_model, device, nepochs=100, lr=0.05, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, fix_bn=False,
                 eval_on_train=False, exem_batch_size=128, logger: ExperimentLogger = None, exemplars_dataset: ExemplarsDataset = None):
        self.server_model = global_server_model
        self.client_model = global_client_model
        
        self.device = device
        self.nepochs = nepochs
        self.lr = lr
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad
        self.momentum = momentum
        self.wd = wd
        self.multi_softmax = multi_softmax
        self.logger = logger
        self.exemplars_dataset = exemplars_dataset
        self.fix_bn = fix_bn
        self.eval_on_train = eval_on_train
        self.optimizer = None
        self.exem_batch_size = exem_batch_size

    def train_loop(self, t, client_loaders, client_models):
        """Contains the epochs loop"""
        lr = self.lr
        best_loss = np.inf
        patience = self.lr_patience
        best_server_model = self.server_model.get_copy()
        best_client_model = self.client_model.get_copy()
        num_clients = len(client_models)
        self.optimizer_server = self._get_server_optimizer()
        
        # Loop epochs
        for e in range(self.nepochs):
            clock0 = time.time()
            
            # Train one epoch
            completed_client = self.train_epoch(t, client_loaders, client_models, lr)
            clock1 = time.time()
            print(' Epoch: {} | Train: time={:5.1f}s |'.format(e+1, clock1 - clock0), end='')
            
            # Validation
            clock3 = time.time()
            valid_loss, valid_acc, _ = self.eval(t, client_loaders, completed_client, num_clients)
            clock4 = time.time()
            print(' Valid: time={:5.1f}s loss={:.3f}, TAw acc={:5.1f}% |'.format(
                clock4 - clock3, valid_loss, 100 * valid_acc), end='')
            self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=valid_loss, group="valid")
            self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * valid_acc, group="valid")

            if e == (self.nepochs*0.7)-2 or e == (self.nepochs*0.9)-2:
                lr /= self.lr_factor
                self.optimizer_server.param_groups[0]['lr'] = lr

            self.logger.log_scalar(task=t, iter=e + 1, name="lr", value=lr, group="train")
            print()
            
    def calculate_metrics(self, outputs, targets):
        """Contains the main Task-Aware and Task-Agnostic metrics"""
        # Task-Agnostic Multi-Head
        if self.multi_softmax:
            outputs = [torch.nn.functional.log_softmax(output, dim=1) for output in outputs]
            pred = torch.cat(outputs, dim=1).argmax(1)
        else:
            pred = torch.cat(outputs, dim=1).argmax(1)
        hits_tag = (pred == targets.to(self.device)).float()
        return 0.0, hits_tag



