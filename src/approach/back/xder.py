# -*- coding: utf-8 -*-
import torch
import warnings
import numpy as np
from copy import deepcopy
from argparse import ArgumentParser
from torch.utils.data import DataLoader
import numpy as np
import random

from .incremental_learning import Inc_Learning_Appr
# from datasets.exemplars_dataset import ExtendedExemplarsDataset
from datasets.exemplars_selection import override_dataset_transform

from xder_utils.spkdloss import SPKDLoss
from torch.nn import functional as F
from utils.args import *
import torch
from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExtendedExemplarsDataset
from xder_utils.augmentations import *
from xder_utils.batch_norm import bn_track_stats
from xder_utils.simclrloss import SupConLoss


# +
class XDerV2(Inc_Learning_Appr):
    NAME = 'xder'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser() -> ArgumentParser:
        parser = ArgumentParser(description='Continual learning via eXtended Dark Experience Replay.')
        add_rehearsal_args(parser)
        parser.add_argument('--alpha', type=float, required=True, help='Penalty weight.')
        parser.add_argument('--beta', type=float, required=True, help='Penalty weight.')
        parser.add_argument('--simclr_temp', type=float, default=5, help='Temperature for SimCLR loss')
        parser.add_argument('--gamma', type=float, default=0.85, help='Weight for logit update')
        parser.add_argument('--simclr_batch_size', type=int, default=64, help='Batch size for SimCLR loss')
        parser.add_argument('--simclr_num_aug', type=int, default=4, help='Number of augmentations for SimCLR loss')
        parser.add_argument('--lambd', type=float, default=0.05, help='Weight for consistency loss')
        parser.add_argument('--constr_eta', type=float, default=0.1, help='Regularization weight for past/future constraints')
        parser.add_argument('--constr_margin', type=float, default=0.3, help='Margin for past/future constraints')
        parser.add_argument('--dp_weight', type=float, default=0, help='Weight for distance preserving loss')
        parser.add_argument('--past_constraint', type=int, default=1, choices=[0, 1], help='Enable past constraint')
        parser.add_argument('--future_constraint', type=int, default=1, choices=[0, 1], help='Enable future constraint')
        parser.add_argument('--align_bn', type=int, default=0, choices=[0, 1], help='Use BatchNorm alignment')
        return parser

#     def __init__(self, model, device, args, transform):
#         super().__init__(model, device, args)
#         self.exemplars_dataset = ExemplarsDataset(transform, None, args.buffer_size)
#         self.update_counter = torch.zeros(self.exemplars_dataset.max_num_exemplars)
#         self.simclr_lss = SupConLoss(temperature=args.simclr_temp, base_temperature=args.simclr_temp, reduction='sum')
#         self.spkdloss = SPKDLoss('batchmean')
        
    def __init__(self, model, device, nepochs=60, lr=0.5, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0.9, wd=1e-5, multi_softmax=False, fix_bn=False,
                 eval_on_train=False, logger=None, exemplars_dataset=None, lamb=1):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, fix_bn, eval_on_train, logger,
                                   exemplars_dataset)
        self.model_old = None
        self.lamb = lamb
        
        self.update_counter = torch.zeros(self.exemplars_dataset.max_num_exemplars)
        self.simclr_lss = SupConLoss(temperature=args.simclr_temp, base_temperature=args.simclr_temp, reduction='sum')
        self.spkdloss = SPKDLoss('batchmean')
        

        # iCaRL is expected to be used with exemplars. If needed to be used without exemplars, overwrite here the
        # `_get_optimizer` function with the one in LwF and update the criterion
        have_exemplars = self.exemplars_dataset.max_num_exemplars + self.exemplars_dataset.max_num_exemplars_per_class
        if not have_exemplars:
            warnings.warn("Warning: xDER is expected to use exemplars. Check documentation.")

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.opt.zero_grad()
        with bn_track_stats(self, self.args.align_bn == 0 or self.current_task == 0):
            outputs = self.model(inputs)

        loss_stream = self.loss(outputs[:, self.n_past_classes:self.n_seen_classes], labels - self.n_past_classes)
        loss_der, loss_derpp = torch.tensor(0.), torch.tensor(0.)
        
        if not self.exemplars_dataset.is_empty():
            buf_idx1, buf_inputs1, buf_labels1, buf_logits1, buf_tl1 = self.exemplars_dataset.get_data(
                self.args.minibatch_size, transform=self.transform, return_index=True, device=self.device)

            if self.args.align_bn:
                buf_inputs1 = torch.cat([buf_inputs1, inputs[:self.args.minibatch_size // self.current_task]])

            buf_outputs1 = self.model(buf_inputs1)
            
            if self.args.align_bn:
                buf_inputs1 = buf_inputs1[:self.args.minibatch_size]
                buf_outputs1 = buf_outputs1[:self.args.minibatch_size]

            mse = F.mse_loss(buf_outputs1, buf_logits1, reduction='none')
            loss_der = self.args.alpha * mse.mean()

            buf_idx2, buf_inputs2, buf_labels2, buf_logits2, buf_tl2 = self.exemplars_dataset.get_data(
                self.args.minibatch_size, transform=self.transform, return_index=True, device=self.device)
            
            with bn_track_stats(self, self.args.align_bn == 0):
                buf_outputs2 = self.model(buf_inputs2)

            buf_ce = self.loss(buf_outputs2[:, :self.n_past_classes], buf_labels2)
            loss_derpp = self.args.beta * buf_ce

            buf_idx = torch.cat([buf_idx1, buf_idx2])
            buf_inputs = torch.cat([buf_inputs1, buf_inputs2])
            buf_labels = torch.cat([buf_labels1, buf_labels2])
            buf_logits = torch.cat([buf_logits1, buf_logits2])
            buf_outputs = torch.cat([buf_outputs1, buf_outputs2])
            buf_tl = torch.cat([buf_tl1, buf_tl2])

            eyey = torch.eye(self.exemplars_dataset.max_num_exemplars).to(buf_idx.device)[buf_idx]
            umask = (eyey * eyey.cumsum(0)).sum(1) < 2

            buf_idx = buf_idx[umask].to(self.exemplars_dataset.device)
            buf_inputs = buf_inputs[umask]
            buf_labels = buf_labels[umask]
            buf_logits = buf_logits[umask]
            buf_outputs = buf_outputs[umask]
            buf_tl = buf_tl[umask]

            with torch.no_grad():
                chosen = ((buf_labels // self.cpt) < self.current_task).to(self.exemplars_dataset.device)
                c = chosen.clone()
                self.update_counter[buf_idx[chosen]] += 1
                chosen[c] = torch.rand_like(chosen[c].float()) * self.update_counter[buf_idx[c]] < 1

                if chosen.any():
                    assert self.current_task > 0
                    to_transplant = self.update_logits(buf_logits[chosen], buf_outputs[chosen], buf_labels[chosen], self.current_task, self.n_tasks - self.current_task)
                    self.exemplars_dataset.logits[buf_idx[chosen], :] = to_transplant.to(self.exemplars_dataset.device)
                    self.exemplars_dataset.task_labels[buf_idx[chosen]] = self.current_task

        loss = loss_stream + loss_der + loss_derpp
        loss.backward()
        self.opt.step()

        return loss.item()

    def end_task(self, dataset):
        self.train()
        if self.current_task > 0:
            examples_per_class = self.args.buffer_size // ((self.current_task + 1) * self.cpt)
            buf_x, buf_lab, buf_log, buf_tl = self.exemplars_dataset.get_all_data()
            self.exemplars_dataset.empty()
            for tl in buf_lab.unique():
                idx = tl == buf_lab
                ex, lab, log, tasklab = buf_x[idx], buf_lab[idx], buf_log[idx], buf_tl[idx]
                first = min(ex.shape[0], examples_per_class)
                self.exemplars_dataset.add_data(
                    examples=ex[:first],
                    labels=lab[:first],
                    logits=log[:first],
                    task_labels=tasklab[:first]
                )

        examples_last_task = self.exemplars_dataset.max_num_exemplars - self.exemplars_dataset.num_seen_examples
        examples_per_class = examples_last_task // self.cpt
        ce = torch.tensor([examples_per_class] * self.cpt).int()
        ce[torch.randperm(self.cpt)[:examples_last_task - (examples_per_class * self.cpt)]] += 1

        with torch.no_grad():
            with bn_track_stats(self, False):
                for data in dataset.train_loader:
                    inputs, labels, not_aug_inputs = data
                    inputs = inputs.to(self.device)
                    not_aug_inputs = not_aug_inputs.to(self.device)
                    outputs = self.global_client_model(inputs)
                    if all(ce == 0):
                        break

                    if self.current_task > 0:
                        outputs = self.update_logits(outputs, outputs, labels, 0, self.current_task)

                    flags = torch.zeros(len(inputs)).bool()
                    for j in range(len(flags)):
                        if ce[labels[j] % self.cpt] > 0:
                            flags[j] = True
                            ce[labels[j] % self.cpt] -= 1

                    self.exemplars_dataset.add_data(
                        examples=not_aug_inputs[flags],
                        labels=labels[flags],
                        logits=outputs.data[flags],
                        task_labels=(torch.ones(self.args.batch_size) * self.current_task)[flags]
                    )

                buf_idx, buf_inputs, buf_labels, buf_logits, _ = self.exemplars_dataset.get_data(
                    self.exemplars_dataset.max_num_exemplars, transform=self.transform, return_index=True, device=self.device)

                buf_outputs = []
                while len(buf_inputs):
                    buf_outputs.append(self.global_client_model(buf_inputs[:self.args.batch_size]))
                    buf_inputs = buf_inputs[self.args.batch_size:]
                buf_outputs = torch.cat(buf_outputs)

                chosen = ((buf_labels // self.cpt) < self.current_task).to(self.exemplars_dataset.device)

                if chosen.any():
                    to_transplant = self.update_logits(buf_logits[chosen], buf_outputs[chosen], buf_labels[chosen], self.current_task, self.n_tasks - self.current_task)
                    self.exemplars_dataset.logits[buf_idx[chosen], :] = to_transplant.to(self.exemplars_dataset.device)
                    self.exemplars_dataset.task_labels[buf_idx[chosen]] = self.current_task

        self.update_counter = torch.zeros(self.args.buffer_size)
        self.train(self.training)

    def update_logits(self, old, new, gt, task_start, n_tasks=1):
        transplant = new[:, task_start * self.cpt:(task_start + n_tasks) * self.cpt]
        gt_values = old[torch.arange(len(gt)), gt]
        max_values = transplant.max(1).values
        coeff = self.args.gamma * gt_values / max_values
        coeff = coeff.unsqueeze(1).repeat(1, self.cpt * n_tasks)
        mask = (max_values > gt_values).unsqueeze(1).repeat(1, self.cpt * n_tasks)
        transplant[mask] *= coeff[mask]
        old[:, task_start * self.cpt:(task_start + n_tasks) * self.cpt] = transplant
        return old
