# -*- coding: utf-8 -*-
import torch
import warnings
import numpy as np
from copy import deepcopy
from argparse import ArgumentParser
from torch.utils.data import DataLoader
import numpy as np
import random

from .incremental_learning_split import Inc_Learning_Appr
from datasets.exemplars_dataset_split import ExemplarsDataset
from datasets.exemplars_selection_split import override_dataset_transform

import cv2


class Appr(Inc_Learning_Appr):
    def __init__(self, server_model, client_model, device, nepochs=60, lr=0.5, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0.9, wd=1e-5, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False,
                 eval_on_train=False, exem_batch_size=128, logger=None, exemplars_dataset=None, lamb=1):
        super(Appr, self).__init__(server_model, client_model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, exem_batch_size, logger,
                                   exemplars_dataset)
        self.server_model_old = None
        self.client_model_old = None
        self.total_exem_info = []
        self.prev_classes = []
        
        have_exemplars = self.exemplars_dataset.max_num_exemplars + self.exemplars_dataset.max_num_exemplars_per_class
        if not have_exemplars:
            warnings.warn("Warning: PRCSL is expected to use exemplars. Check documentation.")
            
        self.nepochs_finetuning = 20
        self._after_train = False
        
    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--lamb', default=1, type=float, required=False,
                            help='Forgetting-intransigence trade-off (default=%(default)s)')
        return parser.parse_known_args(args)

    
    def classify(self, task, features, targets):
        """NME Classification"""
        # Expand means to all batch images
        means = torch.stack(self.exemplar_means)
        means = torch.stack([means] * features.shape[0])
        means = means.transpose(1, 2)
        # Expand all features to all classes
        features = features / features.norm(dim=1).view(-1, 1)
        features = features.unsqueeze(2)
        features = features.expand_as(means)
        # Get distances for all images to all exemplar class means -- nearest prototype
        dists = (features - means).pow(2).sum(1).squeeze()
        if len(targets) == 1:
            dists = dists.unsqueeze(0)
        # Task-Aware Multi-Head
        num_cls = self.server_model.task_cls[task]
        offset = self.server_model.task_offset[task]
        pred = dists[:, offset:offset + num_cls].argmin(1)
        hits_taw = (pred + offset == targets.to(self.device)).float()
        # Task-Agnostic Multi-Head
        pred = dists.argmin(1)
        hits_tag = (pred == targets.to(self.device)).float()
        return hits_taw, hits_tag

    def compute_mean_of_exemplars(self, trn_loader, transform, fix_prev):
        """Change transforms to evaluation for this calculation"""
        with override_dataset_transform(self.exemplars_dataset, transform) as _ds:
            # Change dataloader so it can be fixed to go sequentially (shuffle=False), this allows to keep same order
            icarl_loader = DataLoader(_ds, batch_size=trn_loader.batch_size, shuffle=False,
                                      num_workers=trn_loader.num_workers, pin_memory=trn_loader.pin_memory)
            extracted_features = []
            extracted_targets = []
            with torch.no_grad():
                self.server_model.eval()
                self.client_model.eval()
                for images, targets in icarl_loader:
                    client_feats = self.client_model(images.to(self.device))
                    feats = self.server_model(client_feats, return_features=True)[1]
                    # Normalize(L2-normalization)
                    extracted_features.append(feats / feats.norm(dim=1).view(-1, 1))
                    extracted_targets.extend(targets)
            extracted_features = torch.cat(extracted_features)
            extracted_targets = np.array(extracted_targets)
            for curr_cls in np.unique(extracted_targets):
                if (curr_cls in self.prev_classes) and fix_prev:
                    continue
                # Get all indices from current class
                cls_ind = np.where(extracted_targets == curr_cls)[0]
                # Get all extracted features for current class
                cls_feats = extracted_features[cls_ind]
                # Add the exemplars to the set and normalize
                cls_feats_mean = cls_feats.mean(0) / cls_feats.mean(0).norm()
                self.exemplar_means.append(cls_feats_mean)
                
    def after_train(self, t, client_loaders, client_models):
        """Align D^t and DP exemplars via MIA"""
        self._after_train = True
        orig_lr = self.lr
        orig_nepochs = self.nepochs
        
        self.lr = orig_lr * self.lr_finetune_factor
        self.nepochs = self.nepochs_finetuning
        
        n_client_loaders = len(client_loaders)
        new_client_dataloader = deepcopy(client_loaders)
        class_nums = sum(self.server_model.task_cls[:t+1])
        if self.data_balance_class or self.data_balance_random:
            for i in range(n_client_loaders):
                client_train_dataset = new_client_dataloader[i][0].dataset
                client_train_dataset_labels = client_train_dataset.labels
                
                if self.data_balance_random:
                    client_indices = torch.randperm(len(client_train_dataset))
                    client_train_dataset.images = list(np.array(client_train_dataset.images)\
                                                       [client_indices[:len(self.exemplars_dataset)//n_client_loaders]])
                    client_train_dataset.labels = list(np.array(client_train_dataset.labels)\
                                                       [client_indices[:len(self.exemplars_dataset)//n_client_loaders]])
                        
                print(f"{i} client data, label len: ", len(new_client_dataloader[i][0].dataset.images), len(new_client_dataloader[i][0].dataset.labels))
        
        super().train_loop(t, new_client_dataloader, client_models)
        
        self.nepochs = orig_nepochs
        self.lr = orig_lr
        self._after_train = False

    def train_loop(self, t, client_loaders, client_models):
        """Contains the epochs loop"""
        if t==0:
            self.exemplar_means = []
        else:
            if not self.fix_prev:
                print("Previous class exemplar means: Default")
                self.exemplar_means = []
            else:
                print("Previous class exemplar means: Fix")
            
            # Split exemplar dataset by client numbers
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

        # Training - Step1.Split Learning
        super().train_loop(t, client_loaders, client_models)

        # Collect exemplars
        self.exemplars_dataset.collect_exemplars(t, self.server_model, self.client_model, \
                                                 self.server_model.task_offset, client_loaders, \
                                                 client_loaders[0][1].dataset.transform,\
                                                 dp=True, prev_cls=self.prev_classes, fix_prev=self.fix_prev)
        
        # Step2 A. Constructing DP exemplars
        # Compute differentially private mean on a per-class basis
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
                    dp_image = (mean_image + torch.tensor(np.random.laplace(loc=0, scale=255/(self.dp_mean_batch*self.epsilon), \
                                                                                      size=(h,w,c)), dtype=torch.float32).detach()).numpy()
                else:
                    h,w = mean_image.shape
                    dp_image = (mean_image + torch.tensor(np.random.laplace(loc=0, scale=255/(self.dp_mean_batch*self.epsilon), \
                                                                                      size=(h,w)), dtype=torch.float32).detach()).numpy()
                del mean_image
                dp_image = torch.clamp(torch.from_numpy(dp_image), min=0, max=255).numpy().astype(np.uint8)
                self.exemplars_dataset.images.append(dp_image)
                self.exemplars_dataset.labels.append(l)
                del dp_image
        
        # Save previous classes info
        self.prev_classes=deepcopy(np.unique(self.exemplars_dataset.labels))
        
        # Split exemplar dataset by client numbers
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
        
        # Step2 B. Aligning D^t and DP exemplars via MIA
        self.after_train(t, client_loaders, client_models)
        
        # Step2 C. Computing Class Prototypes
        self.compute_mean_of_exemplars(client_loaders[0][0], client_loaders[0][1].dataset.transform, fix_prev=self.fix_prev)

    def post_train_process(self, t, client_loaders):
        """Runs after training all the epochs of the task (after the train session)"""
        self.server_model_old = deepcopy(self.server_model)
        self.server_model_old.eval()
        self.server_model_old.freeze_all()

        self.client_model_old = deepcopy(self.client_model)
        self.client_model_old.eval()
        self.client_model_old.freeze_all()

    def train_epoch(self, t, client_loaders, client_models, lr):
        """Runs a single epoch"""
        self.server_model.train()
        if t > 0:
            current_client_num = len(client_models)+len(self.exemplar_loaders_split)
        else:
            current_client_num = len(client_models)
            
        j=0
        k=0
        for i in range(current_client_num):
            client = client_models[0]
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
                old_outputs = None
                if t > 0:
                    client_old_outputs = self.client_model_old(images.to(self.device))
                    old_outputs = self.server_model_old(client_old_outputs)

                # Forward current model
                client_outputs = client(images.to(self.device))
                client_fx = client_outputs.clone().detach().requires_grad_(True)
                outputs = self.server_model(client_fx)
                loss = self.criterion(t, outputs, targets.to(self.device), old_outputs)
                del outputs
                if t > 0:
                    del old_outputs
                
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


    def evaluate_server(self, t, fx_client, y, old_outputs):
        """Server-side functions associated with Testing"""
        self.server_model.to(self.device)
        self.server_model.eval()

        with torch.no_grad():
            fx_client = fx_client.to(self.device)
            y = y.to(self.device) 
            fx_server = self.server_model(fx_client)

            # Calculate loss
            loss = self.criterion(t, fx_server, y, old_outputs)
            
            # Calculate accuracy
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
                    # Forward old model
                    old_outputs = None
                    if t > 0:
                        client_old_outputs = self.client_model_old(images.to(self.device))
                        old_outputs = self.server_model_old(client_old_outputs)

                    # Forward current model
                    client_outputs = completed_client(images.to(self.device))

                    if not test: # When validation
                        valid_loss, valid_acc_aw, valid_acc_ag = self.evaluate_server(t, client_outputs, targets.to(self.device), \
                                                                                      old_outputs)
                        total_valid_loss += valid_loss
                        total_valid_acc_aw += valid_acc_aw
                        total_valid_acc_ag += valid_acc_ag
                        iteration += 1
                    else: # When test
                        self.server_model.eval()
                        outputs, feats = self.server_model(client_outputs, return_features=True)
                        loss = self.criterion(t, outputs, targets.to(self.device), old_outputs)
                        hits_taw, hits_tag = self.classify(t, feats, targets)

                        # Log
                        total_loss += loss.item() * len(targets)
                        total_acc_taw += hits_taw.sum().item()
                        total_acc_tag += hits_tag.sum().item()
                        total_num += len(targets)
        
        if not test: # When validation
            final_valid_loss = total_valid_loss / iteration
            final_valid_acc_aw = total_valid_acc_aw / iteration
            final_valid_acc_ag = total_valid_acc_ag / iteration
            return final_valid_loss, final_valid_acc_aw, final_valid_acc_ag
        
        else: # When test
            total_loss += loss.item() * len(targets)
            total_acc_taw += hits_taw.sum().item()
            total_acc_tag += hits_tag.sum().item()
            total_num += len(targets) 
            return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

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

    def criterion(self, t, outputs, targets, old_outputs):
        """Returns the loss value"""
        # Classification loss for new classes
        loss = torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
        
        # Distillation loss for old classes
        if t > 0:
            loss += self.lamb * self.cross_entropy(torch.cat(outputs[:t], dim=1),
                                                   torch.cat(old_outputs[:t], dim=1), exp=1.0 / 2)
        return loss
