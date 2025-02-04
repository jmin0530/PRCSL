# -*- coding: utf-8 -*-
import torch
import warnings
import numpy as np
from copy import deepcopy
from argparse import ArgumentParser
from torch.utils.data import DataLoader

from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset
from datasets.exemplars_selection import override_dataset_transform

from torch.nn import functional as F


# +
class Appr(Inc_Learning_Appr):
    """Class implementing the Incremental Classifier and Representation Learning (iCaRL) approach
    described in https://arxiv.org/abs/1611.07725
    Original code available at https://github.com/srebuffi/iCaRL
    """

    def __init__(self, model, device, nepochs=60, lr=0.5, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0.9, wd=1e-5, multi_softmax=False, fix_bn=False,
                 eval_on_train=False, logger=None, exemplars_dataset=None, lamb=1):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, fix_bn, eval_on_train, logger, exemplars_dataset)
        self.model_old = None
        self.lamb = lamb
        self.alpha = 0.5
        self.num_seen_examples = 0

        # iCaRL is expected to be used with exemplars. If needed to be used without exemplars, overwrite here the
        # `_get_optimizer` function with the one in LwF and update the criterion
        have_exemplars = self.exemplars_dataset.max_num_exemplars + self.exemplars_dataset.max_num_exemplars_per_class
        self.exemplars_dataset.logits = []
        self.exemplars_dataset.return_logits = True
        if not have_exemplars:
            warnings.warn("Warning: iCaRL is expected to use exemplars. Check documentation.")

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
            
#             trn_buff_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
#                                                      batch_size=trn_loader.batch_size,
#                                                      shuffle=True,
#                                                      num_workers=trn_loader.num_workers,
#                                                      pin_memory=trn_loader.pin_memory)
            

        # FINETUNING TRAINING -- contains the epochs loop
        super().train_loop(t, trn_loader, val_loader)

#         if t > 0:
#             self.exemplars_dataset.collect_exemplars(self.model, trn_buff_loader, val_loader.dataset.transform)
#         else:
#             self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)
        
        # add logits to memory
        self.exemplars_dataset.add_logits_to_memory(t, self.exemplars_dataset, self.model, trn_loader.batch_size, self.device)

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
            loss = self.criterion(t, outputs, targets.to(self.device), False)
            loss.backward()
            
            # exemplar에서 랜덤으로 batchsize 만큼 뽑은 후 loader 씌워서 augmentation 적용
            # 그 후 현재 모델에 exemplar data 입력하여 나온 logit값(outputs_buff)와
            # exemplar에 저장된 logit값(logits_buff) 간 mse loss 계산
            if self.exemplars_dataset.images != []: 
                choice = list(np.random.choice(len(self.exemplars_dataset.images), size=trn_loader.batch_size, replace=False))
                batch_exemplars = deepcopy(self.exemplars_dataset)
#                 print("batch_exemplars.images type: ", type(batch_exemplars.images)) # list
#                 print("batch_exemplars.images[0] type: ", type(batch_exemplars.images[0])) # torch.tensor
#                 exit(True)
#                 batch_exemplars.images = torch.tensor(batch_exemplars.images)[choice]
                batch_exemplars.images = [batch_exemplars.images[i] for i in choice]
                print(len(batch_exemplars.images))
                exit(True)
#                 batch_exemplars.labels = torch.tensor(batch_exemplars.labels)[choice]
                batch_exemplars.labels = [batch_exemplars.labels[i] for i in choice]
#                 batch_exemplars.logits = torch.tensor(batch_exemplars.logits)[choice]
                batch_exemplars.logits = [batch_exemplars.logits[i] for i in choice]
                batch_exemplars.return_logits = True
                buff_loader = torch.utils.data.DataLoader(batch_exemplars,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=False,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory)
                
                for buff_images, _, buff_logits, _ in buff_loader:
                    outputs_buff = self.model(buff_images.to(self.device))
                loss_mse = self.criterion(t, outputs_buff, buff_logits.to(self.device), True)
                loss_mse.backward()
#             else:#######
#                 self.add_exemplar_data(no_aug_images, logits = outputs[0].data, labels = targets.data)
#                 attr = [('images',no_aug_images.shape), ('labels', targets.shape), ('logits', outputs[0].shape)]
#                 for att in attr:
#                     typ = torch.int64 if att[0].endswith('els') else torch.float32
#                     setattr(self.exemplars_dataset, att[0], 
#                             torch.zeros((self.exemplars_dataset.max_num_exemplars, *att[1][1:]), dtype=typ, device=self.device))
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()
            
            # reservoir sampling
            # not_aug_inputs:augmentation이 적용 안된 배치단위 입력 이미지
#             print("not_aug_inputs: ", type(no_aug_images))
            self.add_exemplar_data(no_aug_images, logits = outputs[0].data, labels = targets.data)
             

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
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num += len(targets)
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    # Algorithm 3: classification and distillation terms -- original formulation has no trade-off parameter (lamb=1)
    def criterion(self, t, outputs, targets, now_exem=False):
        """Returns the loss value"""
        if not now_exem:
            # Classification loss for new classes
            loss = torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
        else:
            loss = self.alpha * F.mse_loss(torch.cat(outputs[:t], dim=1), targets)
        return loss
    
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
    
    def add_exemplar_data(self, examples, logits = None, labels = None):
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        """
        for i in range(examples.shape[0]):
            index = self.reservoir(self.num_seen_examples, self.exemplars_dataset.max_num_exemplars)
            self.num_seen_examples += 1
            if self.num_seen_examples < self.exemplars_dataset.max_num_exemplars:
                self.exemplars_dataset.images.append(examples[i].to(self.device))
#                 if self.exemplars_dataset.labels is not None:
                self.exemplars_dataset.labels.append(labels[i].to(self.device))
#                 if self.exemplars_dataset.logits is not None:
                self.exemplars_dataset.logits.append(logits[i].to(self.device))
            else:
                if index >= 0:
                    # t = 0에서 IndexError: list assignment index out of range
                    # 원인: exemplar가 비어있는 상태에서 indexing 할려다보니 오류 발생한 것임
                    self.exemplars_dataset.images[index] = examples[i].to(self.device) 
#                     if self.exemplars_dataset.labels is not None:
                    self.exemplars_dataset.labels[index] = labels[i].to(self.device)
#                     if self.exemplars_dataset.logits is not None:
                    self.exemplars_dataset.logits[index] = logits[i].to(self.device)

# +
# import torch
# a = torch.tensor([1,2,3,4,5,6,7])
# ind = [0,2,4]
# print(a[ind])
