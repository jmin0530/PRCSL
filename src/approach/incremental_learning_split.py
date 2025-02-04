# -*- coding: utf-8 -*-
import time
import torch
import numpy as np
from argparse import ArgumentParser
import copy

from loggers.exp_logger import ExperimentLogger
from datasets.exemplars_dataset_split import ExemplarsDataset


class Inc_Learning_Appr:
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
    
    def _get_server_optimizer(self):
        """Returns the optimizer"""
        if self.opt == 'adam':
            return torch.optim.Adam(self.server_model.parameters(), lr=self.lr, weight_decay=self.wd)
        elif self.opt == 'sgd':
            return torch.optim.SGD(self.server_model.parameters(), lr=self.lr, weight_decay=self.wd, momentum=self.momentum)
    
    def train(self, t, client_loaders, taskcla):
        # Global client model initialization
        client_models = []
        for i in range(len(client_loaders)):
            client_models.append(self.client_model.to(self.device))

        self.train_loop(t, client_loaders, client_models)
        self.post_train_process(t, client_loaders)

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

            # Update best validation loss
            if valid_loss < best_loss:
                best_loss = valid_loss
                patience = self.lr_patience
                best_server_model = self.server_model.get_copy()
                best_client_model = completed_client.get_copy()
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
        self.server_model.set_state_dict(best_server_model)
        self.client_model.set_state_dict(best_client_model)

    def post_train_process(self, t, client_loader):
        """Runs after training all the epochs of the task (after the train session)"""
        pass

    def train_epoch(self, t, client_loaders, client_models, lr):
        """Runs a single epoch"""
        current_client_num = len(client_models)
        for i in range(current_client_num):
            # Load a client
            client = client_models[i]
            client.train()
            
            # Receive previous client's weight
            if i > 0:
                client.load_state_dict(previous_client.state_dict())
                
            if self.opt == 'adam':
                client_optim = torch.optim.Adam(client.parameters(), lr=lr, weight_decay=self.wd)
            elif self.opt == 'sgd':
                client_optim = torch.optim.SGD(client.parameters(), lr=lr, weight_decay=self.wd, momentum=self.momentum)

            loader = client_loaders[i][0]
            for images, targets in loader: 
                # Forward current client
                client_outputs = client(images.to(self.device))
                client_fx = client_outputs.clone().detach().requires_grad_(True)

                # Training server
                # Current client sends it's smashed data to the server
                # If server training is done, current client receives the smashed data gradient
                dfx = self.train_server(t, client_fx, targets.to(self.device))
                
                # Client backpropagation
                client_optim.zero_grad()
                client_outputs.backward(dfx)
                client_optim.step()
                
            if i == len(client_models)-1:
                return client
            else:
                previous_client = copy.deepcopy(client).to(self.device)
                
    def train_server(self, t, fx_client, y):
        """Server-side training"""
        self.server_model.train()
        self.optimizer_server.zero_grad()
        fx_client = fx_client.to(self.device)
        y = y.to(self.device)
        
        # Forward propagation
        fx_server = self.server_model(fx_client)

        # Calculate loss
        loss = self.criterion(t, fx_server, y)

        # Backpropagation
        loss.backward()
        dfx_client = fx_client.grad.clone().detach()
        self.optimizer_server.step()
        
        return dfx_client

    def evaluate_server(self, t, fx_client, y):
        """Server-side evaluation"""
        self.server_model.to(self.device)
        self.server_model.eval()

        with torch.no_grad():
            fx_client = fx_client.to(self.device)
            y = y.to(self.device) 
            # Forward propagation
            fx_server = self.server_model(fx_client)

            # Calculate loss
            loss = self.criterion(t, fx_server, y)
            
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            hits_taw, hits_tag = self.calculate_metrics(fx_server, y)
            
            # Accumulate total loss and total accuracies
            total_loss += loss.item() * len(y)
            total_acc_taw += hits_taw.sum().item()
            total_acc_tag += hits_tag.sum().item()
            total_num += len(y)
                
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    
    def eval(self, t, client_loaders, completed_client, num_clients, test=False):
        """Evaluation"""
        total_valid_loss, total_valid_acc_aw, total_valid_acc_ag = 0.0, 0.0, 0.0
        iteration = 0
        for i in range(num_clients):
            if test:
                loader = client_loaders[i][2]
            else:
                loader = client_loaders[i][1]
                
            with torch.no_grad():
                completed_client.eval()
                self.server_model.eval()
                for images, targets in loader:
                    # Forward client model
                    client_outputs = completed_client(images.to(self.device))

                    # Forward server model and evaluate
                    valid_loss, valid_acc_aw, valid_acc_ag = self.evaluate_server(t, client_outputs, targets.to(self.device))

                    total_valid_loss += valid_loss
                    total_valid_acc_aw += valid_acc_aw
                    total_valid_acc_ag += valid_acc_ag
                    iteration += 1
        
        final_valid_loss = total_valid_loss / iteration
        final_valid_acc_aw = total_valid_acc_aw / iteration
        final_valid_acc_ag = total_valid_acc_ag / iteration
        
        return final_valid_loss, final_valid_acc_aw, final_valid_acc_ag
          

    def calculate_metrics(self, outputs, targets):
        """Contains the main Task-Aware and Task-Agnostic metrics"""
        pred = torch.zeros_like(targets.to(self.device))
        # Task-Aware Multi-Head
        for m in range(len(pred)):
            this_task = (self.server_model.task_cls.cumsum(0).to(self.device) <= targets[m]).sum()
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

    def criterion(self, t, outputs, targets):
        """Returns the loss value"""
        return torch.nn.functional.cross_entropy(outputs[t], targets - self.server_model.task_offset[t])


