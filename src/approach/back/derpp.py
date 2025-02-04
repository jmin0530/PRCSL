# -*- coding: utf-8 -*-
import torch
import warnings
import numpy as np
from copy import deepcopy
from argparse import ArgumentParser
from torch.utils.data import DataLoader

from .incremental_learning_der import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset
from datasets.exemplars_selection import override_dataset_transform

from torch.nn import functional as F

import torchvision.transforms as transforms

import matplotlib.pyplot as plt

from .augmentations.kornia_utils import to_kornia_transform, apply_transform


class Appr(Inc_Learning_Appr):
    def __init__(self, model, device, nepochs=60, lr=0.5, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0.9, wd=1e-5, multi_softmax=False, fix_bn=False,
                 eval_on_train=False, logger=None, exemplars_dataset=None, lamb=1):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, fix_bn, eval_on_train, logger, exemplars_dataset)
        self.model_old = None
        self.lamb = lamb
        self.num_seen_examples = 0

        # DER++ is expected to be used with exemplars. If needed to be used without exemplars, overwrite here the
        # `_get_optimizer` function with the one in LwF and update the criterion
        have_exemplars = self.exemplars_dataset.max_num_exemplars + self.exemplars_dataset.max_num_exemplars_per_class
        buff_transform = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))])
        # prcsl norm: (0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)
        # mammoth norm: (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        self.buffer_transform = to_kornia_transform(buff_transform.transforms)
        if not have_exemplars:
            warnings.warn("Warning: DER is expected to use exemplars. Check documentation.")

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        # Sec. 4. " allowing iCaRL to balance between CE and distillation loss."
        parser.add_argument('--lamb', default=1, type=float, required=False,
                            help='Forgetting-intransigence trade-off (default=%(default)s)')
        return parser.parse_known_args(args)

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""
        trn_loader = torch.utils.data.DataLoader(trn_loader.dataset,
                                                 batch_size=trn_loader.batch_size,
                                                 shuffle=True,
                                                 num_workers=trn_loader.num_workers,
                                                 pin_memory=trn_loader.pin_memory)
        
        self.buffer_batch_size = trn_loader.batch_size

        # FINETUNING TRAINING -- contains the epochs loop
        super().train_loop(t, trn_loader, val_loader)
        
        print("Buffer data labels(length): ", self.exemplars_dataset.labels, f"({len(self.exemplars_dataset.images)})")

    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""

        # Save old model to extract features later. This is different from the original approach, since they propose to
        #  extract the features and store them for future usage. However, when using data augmentation, it is easier to
        #  keep the model frozen and extract the features when needed.
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        for i, (images, targets, no_aug_images) in enumerate(trn_loader):
            self.optimizer.zero_grad()
            # Forward current model
            outputs = self.model(images.to(self.device))
            loss = self.criterion(t, outputs, targets.to(self.device))
            loss.backward()
            
            if self.exemplars_dataset.images != []:
                # reg 1
                buff_images, _, buff_logits = self.get_buff_data()
                outputs_buff = self.model(buff_images)
                loss_mse = 0.1 * F.mse_loss(outputs_buff[0], buff_logits)
                loss_mse.backward()
                
                # reg 2
                buff_images, buff_labels, _ = self.get_buff_data()
                outputs_buff = self.model(buff_images)
                loss_ce = 0.5 * self.criterion(t, outputs_buff, buff_labels)
                loss_ce.backward()
#             torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

            self.add_exemplar_data(no_aug_images, logits = outputs[0].data, labels = targets)
            
    def get_buff_data(self):
        num_avail_samples = len(self.exemplars_dataset.images)
        num_avail_samples = min(self.num_seen_examples, num_avail_samples)
        if self.buffer_batch_size > min(num_avail_samples, len(self.exemplars_dataset.images)):
            size = min(num_avail_samples, len(self.exemplars_dataset.images))
        else:
            size = self.buffer_batch_size
        choice = list(np.random.choice(num_avail_samples, size=size, replace=False))
        buff_images = apply_transform(deepcopy(self.exemplars_dataset.images[choice]), self.buffer_transform, True)
        buff_labels = deepcopy(self.exemplars_dataset.labels[choice])
        buff_logits = deepcopy(self.exemplars_dataset.logits[choice])
        
        return buff_images, buff_labels, buff_logits
            
    def reservoir(self, num_seen_examples: int, buffer_size: int) -> int:
        """
        Reservoir sampling algorithm.

        Args:
            num_seen_examples: the number of seen examples
            buffer_size: the maximum buffer size

        Returns:
            the target index if the current image is sampled, else -1
        """
        if num_seen_examples < buffer_size:
            return num_seen_examples

        rand = np.random.randint(0, num_seen_examples + 1)
        if rand < buffer_size:
            return rand
        else:
            return -1
            
    def add_exemplar_data(self, images, logits = None, labels = None):
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        """
        if len(self.exemplars_dataset.images) == 0:
            for attr_str in ['images', 'logits', 'labels']:
                attr = eval(attr_str)
                if attr_str == 'labels':
                    tensor_type = torch.long
                else:
                    tensor_type = torch.float32
                setattr(self.exemplars_dataset, attr_str, torch.zeros((self.exemplars_dataset.max_num_exemplars,
                        *attr.shape[1:]), dtype=tensor_type, device=self.device))
                
        for i in range(images.shape[0]):
            index = self.reservoir(self.num_seen_examples, self.exemplars_dataset.max_num_exemplars)
            self.num_seen_examples += 1
            if index >= 0:
                self.exemplars_dataset.images[index] = images[i].to(self.device)
                self.exemplars_dataset.labels[index] = labels[i].to(self.device)
                self.exemplars_dataset.logits[index] = logits[i].to(self.device)

    def eval(self, t, val_loader):
        """Contains the evaluation code"""
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()
            for images, targets, _ in val_loader:
                # Forward current model
                outputs, feats = self.model(images.to(self.device), return_features=True)
                loss = self.criterion(t, outputs, targets.to(self.device))
                # during training, the usual accuracy is computed on the outputs
                hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                # Log
                total_loss += loss.item() * len(targets)
#                 total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num += len(targets)
        return total_loss / total_num, 0.0, total_acc_tag / total_num

    # Algorithm 3: classification and distillation terms -- original formulation has no trade-off parameter (lamb=1)
    def criterion(self, t, outputs, targets):
        """Returns the loss value"""
        # Classification loss for new classes
        loss = torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
        return loss
    
    def calculate_metrics(self, outputs, targets):
        """Contains the main Task-Aware and Task-Agnostic metrics"""
#         pred = torch.zeros_like(targets.to(self.device))
#         # Task-Aware Multi-Head
#         for m in range(len(pred)):
#             this_task = (self.model.task_cls.cumsum(0) <= targets[m]).sum()
#             pred[m] = outputs[this_task][m].argmax() + self.model.task_offset[this_task]
#         hits_taw = (pred == targets.to(self.device)).float()
        # Task-Agnostic Multi-Head
        if self.multi_softmax:
            outputs = [torch.nn.functional.log_softmax(output, dim=1) for output in outputs]
            pred = torch.cat(outputs, dim=1).argmax(1)
        else:
            pred = torch.cat(outputs, dim=1).argmax(1)
        hits_tag = (pred == targets.to(self.device)).float()
        return 0.0, hits_tag


def imgsave(inp, i, label, adding=False):
    """tensor를 입력받아 일반적인 이미지로 보여줍니다."""
#     inp = inp.cpu().numpy().transpose((1, 2, 0))
#     mean = np.array([0.5071, 0.4866, 0.4409])
#     std = np.array([0.2009, 0.1984, 0.2023])
#     inp = std * inp + mean
#     inp = np.clip(inp, 0, 1)
    
    inp = inp.permute(1, 2, 0).cpu()
    plt.imshow(inp)
    if adding:
        plt.savefig(f'add_exemplar_data img[{i}] (label: {label})')
    else:
        plt.savefig(f'training no_aug img[{i}] (label: {label})')
