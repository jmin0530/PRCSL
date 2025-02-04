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

from xder_utils.batchnorm import bn_track_stats


# +
class Appr(Inc_Learning_Appr):
    def __init__(self, server_model, client_model, device, nepochs=60, lr=0.5, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0.9, wd=1e-5, multi_softmax=False, fix_bn=False,
                 eval_on_train=False, exem_batch_size=128, logger=None, exemplars_dataset=None, lamb=1):
        super(Appr, self).__init__(server_model, client_model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, fix_bn, eval_on_train, exem_batch_size, logger,
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
                
#     def after_train(self, t, client_loaders, client_models):
#         """Align D^t and DP exemplars via MIA"""
#         self._after_train = True
#         orig_lr = self.lr
#         orig_nepochs = self.nepochs
        
#         self.lr = orig_lr * self.lr_finetune_factor
#         self.nepochs = self.nepochs_finetuning
        
#         n_client_loaders = len(client_loaders)
#         new_client_dataloader = deepcopy(client_loaders)
#         class_nums = sum(self.server_model.task_cls[:t+1])
#         if self.data_balance_class or self.data_balance_random:
#             for i in range(n_client_loaders):
#                 client_train_dataset = new_client_dataloader[i][0].dataset
#                 client_train_dataset_labels = client_train_dataset.labels
                
#                 if self.data_balance_random:
#                     client_indices = torch.randperm(len(client_train_dataset))
#                     client_train_dataset.images = list(np.array(client_train_dataset.images)\
#                                                        [client_indices[:len(self.exemplars_dataset)//n_client_loaders]])
#                     client_train_dataset.labels = list(np.array(client_train_dataset.labels)\
#                                                        [client_indices[:len(self.exemplars_dataset)//n_client_loaders]])
                        
#                 print(f"{i} client data, label len: ", len(new_client_dataloader[i][0].dataset.images), len(new_client_dataloader[i][0].dataset.labels))
        
#         super().train_loop(t, new_client_dataloader, client_models)
        
#         self.nepochs = orig_nepochs
#         self.lr = orig_lr
#         self._after_train = False
    def after_train(t, client_loaders, client_models):
        tng = self.training
        self.train()

        # fdr reduce coreset
        if self.current_task > 0:
            examples_per_class = self.args.buffer_size // ((self.current_task + 1) * self.cpt)
            buf_x, buf_lab, buf_log, buf_tl = self.buffer.get_all_data()
            self.buffer.empty()
            for tl in buf_lab.unique():
                idx = tl == buf_lab
                ex, lab, log, tasklab = buf_x[idx], buf_lab[idx], buf_log[idx], buf_tl[idx]
                first = min(ex.shape[0], examples_per_class)
                self.buffer.add_data(
                    examples=ex[:first],
                    labels=lab[:first],
                    logits=log[:first],
                    task_labels=tasklab[:first]
                )

        # fdr add new task
        examples_last_task = self.buffer.buffer_size - self.buffer.num_seen_examples
        examples_per_class = examples_last_task // self.cpt
        ce = torch.tensor([examples_per_class] * self.cpt).int()
        ce[torch.randperm(self.cpt)[:examples_last_task - (examples_per_class * self.cpt)]] += 1

        with torch.no_grad():
            with bn_track_stats(self, False):
                for data in dataset.train_loader:
                    inputs, labels, not_aug_inputs = data
                    inputs = inputs.to(self.device)
                    not_aug_inputs = not_aug_inputs.to(self.device)
                    outputs = self.net(inputs)
                    if all(ce == 0):
                        break

                    # update past
                    if self.current_task > 0:
                        outputs = self.update_logits(outputs, outputs, labels, 0, self.current_task)

                    flags = torch.zeros(len(inputs)).bool()
                    for j in range(len(flags)):
                        if ce[labels[j] % self.cpt] > 0:
                            flags[j] = True
                            ce[labels[j] % self.cpt] -= 1

                    self.buffer.add_data(examples=not_aug_inputs[flags],
                                         labels=labels[flags],
                                         logits=outputs.data[flags],
                                         task_labels=(torch.ones(self.args.batch_size) * self.current_task)[flags])

                # update future past
                buf_idx, buf_inputs, buf_labels, buf_logits, _ = self.buffer.get_data(self.buffer.buffer_size,
                                                                                      transform=self.transform, return_index=True, device=self.device)

                buf_outputs = []
                while len(buf_inputs):
                    buf_outputs.append(self.net(buf_inputs[:self.args.batch_size]))
                    buf_inputs = buf_inputs[self.args.batch_size:]
                buf_outputs = torch.cat(buf_outputs)

                chosen = ((buf_labels // self.cpt) < self.current_task).to(self.buffer.device)

                if chosen.any():
                    to_transplant = self.update_logits(buf_logits[chosen], buf_outputs[chosen], buf_labels[chosen], self.current_task, self.n_tasks - self.current_task)
                    self.buffer.logits[buf_idx[chosen], :] = to_transplant.to(self.buffer.device)
                    self.buffer.task_labels[buf_idx[chosen]] = self.current_task

        self.update_counter = torch.zeros(self.args.buffer_size)

        self.train(tng)

    def train_loop(self, t, client_loaders, client_models):
        """Contains the epochs loop"""
        if t==0:
            self.exemplar_means = []
        else:
            if not self.fix_prev:
                print("Previous class exemplar means: Dynamic")
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
            
            if t > 0: # xDER
                # Distillation Replay Loss (all heads)
                buf_idx1, buf_inputs1, buf_labels1, buf_logits1, buf_tl1 = self.buffer.get_data(
                    self.args.minibatch_size, transform=self.transform, return_index=True, device=self.device) # self.exemplar 구성 수정 필요

                buf_outputs1 = self.net(buf_inputs1)

                mse = F.mse_loss(buf_outputs1, buf_logits1, reduction='none')
                loss_der = self.args.alpha * mse.mean()

                # Label Replay Loss (past heads)
                buf_idx2, buf_inputs2, buf_labels2, buf_logits2, buf_tl2 = self.buffer.get_data(
                    self.args.minibatch_size, transform=self.transform, return_index=True, device=self.device) # self.exemplar 구성 수정 필요
                with bn_track_stats(self, self.args.align_bn == 0):
                    buf_outputs2 = self.net(buf_inputs2)
                buf_ce = self.loss(buf_outputs2[:, :self.n_past_classes], buf_labels2)
                loss_derpp = self.args.beta * buf_ce

                # Merge Batches & Remove Duplicates
                buf_idx = torch.cat([buf_idx1, buf_idx2])
                buf_inputs = torch.cat([buf_inputs1, buf_inputs2])
                buf_labels = torch.cat([buf_labels1, buf_labels2])
                buf_logits = torch.cat([buf_logits1, buf_logits2])
                buf_outputs = torch.cat([buf_outputs1, buf_outputs2])
                buf_tl = torch.cat([buf_tl1, buf_tl2])

                # remove dupulicates
                eyey = torch.eye(self.buffer.buffer_size).to(buf_idx.device)[buf_idx]
                umask = (eyey * eyey.cumsum(0)).sum(1) < 2

                buf_idx = buf_idx[umask].to(self.buffer.device)
                buf_inputs = buf_inputs[umask]
                buf_labels = buf_labels[umask]
                buf_logits = buf_logits[umask]
                buf_outputs = buf_outputs[umask]
                buf_tl = buf_tl[umask]

                # Update Future Past Logits
                with torch.no_grad():
                    chosen = ((buf_labels // self.cpt) < self.current_task).to(self.buffer.device)
                    c = chosen.clone()
                    self.update_counter[buf_idx[chosen]] += 1
                    chosen[c] = torch.rand_like(chosen[c].float()) * self.update_counter[buf_idx[c]] < 1

                    if chosen.any():
                        assert self.current_task > 0
                        to_transplant = self.update_logits(buf_logits[chosen], buf_outputs[chosen], buf_labels[chosen], self.current_task, self.n_tasks - self.current_task)
                        self.buffer.logits[buf_idx[chosen], :] = to_transplant.to(self.buffer.device)
                        self.buffer.task_labels[buf_idx[chosen]] = self.current_task
                        
            # Consistency Loss (future heads)
            loss_cons, loss_dp = torch.tensor(0.), torch.tensor(0.)
            loss_constr_futu = torch.tensor(0.)
            if t < self.n_tasks - 1:

                scl_labels = labels  # [:self.args.simclr_batch_size]
                scl_na_inputs = not_aug_inputs  # [:self.args.simclr_batch_size]
                if not self.buffer.is_empty():
                    buf_idxscl, buf_na_inputsscl, buf_labelsscl, buf_logitsscl, _ = self.buffer.get_data(self.args.simclr_batch_size,
                                                                                                         transform=None, return_index=True, device=self.device)
                    scl_na_inputs = torch.cat([buf_na_inputsscl, scl_na_inputs])
                    scl_labels = torch.cat([buf_labelsscl, scl_labels])
                with torch.no_grad():
                    scl_inputs = self.gpu_augmentation(scl_na_inputs.repeat_interleave(self.args.simclr_num_aug, 0)).to(self.device)

                with bn_track_stats(self, self.args.align_bn == 0):
                    scl_outputs = self.net(scl_inputs)

                scl_featuresFull = scl_outputs.reshape(-1, self.args.simclr_num_aug, scl_outputs.shape[-1])

                scl_features = scl_featuresFull[:, :, (self.current_task + 1) * self.cpt:]
                scl_n_heads = self.n_tasks - self.current_task - 1

                scl_features = torch.stack(scl_features.split(self.cpt, 2), 1)

                loss_cons = torch.stack([self.simclr_lss(features=F.normalize(scl_features[:, h], dim=2), labels=scl_labels) for h in range(scl_n_heads)]).sum()
                loss_cons /= scl_n_heads * scl_features.shape[0]
                loss_cons *= self.args.lambd

                # DP loss
                if self.args.dp_weight > 0 and not self.buffer.is_empty():
                    dp_features = scl_featuresFull[:len(buf_logitsscl), :, (self.current_task + 1) * self.cpt:]
                    dp_logits = buf_logitsscl[:, (self.current_task + 1) * self.cpt:]

                    dp_features = torch.stack(dp_features.split(self.cpt, 2), 1)

                    dp_logits = torch.stack(dp_logits.split(self.cpt, 1), 1)

                    loss_dp = self.args.dp_weight * torch.mean(torch.stack(
                        [self.spkdloss(dp_features[:, i, k, :], dp_logits[:, i, :]) for i in range(self.n_tasks - self.current_task - 1) for k in range(self.args.simclr_num_aug)]
                    ))

                # Future Logits Constraint
                if self.args.future_constraint:
                    bad_head = outputs[:, (self.current_task + 1) * self.cpt:]
                    good_head = outputs[:, self.current_task * self.cpt:(self.current_task + 1) * self.cpt]

                    if not self.buffer.is_empty():
                        buf_tlgt = buf_labels // self.cpt
                        bad_head = torch.cat([bad_head, buf_outputs[:, (self.current_task + 1) * self.cpt:]])
                        good_head = torch.cat([good_head, torch.stack(buf_outputs.split(self.cpt, 1), 1)[torch.arange(len(buf_tlgt)), buf_tlgt]])

                    loss_constr = bad_head.max(1)[0] + self.args.constr_margin - good_head.max(1)[0]

                    mask = loss_constr > 0
                    if (mask).any():
                        loss_constr_futu = self.args.constr_eta * loss_constr[mask].mean()
            # Past Logits Constraint
            loss_constr_past = torch.tensor(0.).type(loss_stream.dtype)
            if self.args.past_constraint and self.current_task > 0:
                chead = F.softmax(outputs[:, :(self.current_task + 1) * self.cpt], 1)

                good_head = chead[:, self.current_task * self.cpt:(self.current_task + 1) * self.cpt]
                bad_head = chead[:, :self.cpt * self.current_task]

                loss_constr = bad_head.max(1)[0].detach() + self.args.constr_margin - good_head.max(1)[0]

                mask = loss_constr > 0

                if (mask).any():
                    loss_constr_past = self.args.constr_eta * loss_constr[mask].mean()

            loss = loss_stream + loss_der + loss_derpp + loss_cons + loss_dp + loss_constr_futu + loss_constr_past

            loss.backward()
            self.opt.step()
            for images, targets in loader:
                old_outputs = None
                if t > 0:
                    client_old_outputs = self.client_model_old(images.to(self.device))
                    old_outputs = self.server_model_old(client_old_outputs)

                # Forward current model
                client_outputs = client(images.to(self.device))
                client_fx = client_outputs.clone().detach().requires_grad_(True)
                outputs = self.server_model(client_fx)
                
                # Present head
                loss_stream = self.criterion(outputs[:, self.n_past_classes:self.n_seen_classes], labels - self.n_past_classes)
#                 loss = self.criterion(t, outputs, targets.to(self.device), old_outputs)
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
        
        # L_DER(equation 5)
        # buf_outputs1, buf_logits1을 메모리에서 불러오도록 해야 함
        mse = F.mse_loss(buf_outputs1, buf_logits1, reduction='none')
        loss_der = self.args.alpha * mse.mean()
        return loss
