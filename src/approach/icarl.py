import torch
import warnings
import numpy as np
from copy import deepcopy
from argparse import ArgumentParser
from torch.utils.data import DataLoader

from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset
from datasets.exemplars_selection import override_dataset_transform


class Appr(Inc_Learning_Appr):
    """Class implementing the Incremental Classifier and Representation Learning (iCaRL) approach
    described in https://arxiv.org/abs/1611.07725
    Original code available at https://github.com/srebuffi/iCaRL
    """

    def __init__(self, model, device, nepochs=60, lr=0.5, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0.9, wd=1e-5, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False,
                 eval_on_train=False, logger=None, exemplars_dataset=None, lamb=1):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset)
        self.model_old = None
        self.lamb = lamb

        # iCaRL is expected to be used with exemplars. If needed to be used without exemplars, overwrite here the
        # `_get_optimizer` function with the one in LwF and update the criterion
        have_exemplars = self.exemplars_dataset.max_num_exemplars + self.exemplars_dataset.max_num_exemplars_per_class
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

    # Algorithm 1: iCaRL NCM Classify
    def classify(self, task, features, targets):
        # expand means to all batch images
#         print(self.exemplar_means)
        means = torch.stack(self.exemplar_means)
#         print("means: ", means.shape)
        means = torch.stack([means] * features.shape[0])
#         print("means stack: ", means.shape)
        means = means.transpose(1, 2)
#         print("means transpose(1,2): ", means.shape)
        # expand all features to all classes
#         print("features: ", features.shape)
        features = features / features.norm(dim=1).view(-1, 1)
#         print("norm features: ", features.shape)
        features = features.unsqueeze(2)
#         print("features unsqueeze(2): ", features.shape)
        features = features.expand_as(means)
#         print("features expand as means: ", features.shape)
        # get distances for all images to all exemplar class means -- nearest prototype
        dists = (features - means).pow(2).sum(1).squeeze()
#         print("dists: ", dists.shape)
#         exit(True)
        # Task-Aware Multi-Head
        num_cls = self.model.task_cls[task]
        offset = self.model.task_offset[task]
        pred = dists[:, offset:offset + num_cls].argmin(1)
        hits_taw = (pred + offset == targets.to(self.device)).float()
        # Task-Agnostic Multi-Head
        pred = dists.argmin(1)
        hits_tag = (pred == targets.to(self.device)).float()
        return hits_taw, hits_tag

    def compute_mean_of_exemplars(self, trn_loader, transform):
        # change transforms to evaluation for this calculation
        with override_dataset_transform(self.exemplars_dataset, transform) as _ds:
            # change dataloader so it can be fixed to go sequentially (shuffle=False), this allows to keep same order
            icarl_loader = DataLoader(_ds, batch_size=trn_loader.batch_size, shuffle=False,
                                      num_workers=trn_loader.num_workers, pin_memory=trn_loader.pin_memory)
            # extract features from the model for all train samples
            # Page 2: "All feature vectors are L2-normalized, and the results of any operation on feature vectors,
            # e.g. averages are also re-normalized, which we do not write explicitly to avoid a cluttered notation."
            extracted_features = []
            extracted_targets = []
            with torch.no_grad():
                self.model.eval()
                for images, targets in icarl_loader:
                    feats = self.model(images.to(self.device), return_features=True)[1]
                    # normalize
                    extracted_features.append(feats / feats.norm(dim=1).view(-1, 1))
                    extracted_targets.extend(targets)
            extracted_features = torch.cat(extracted_features)
            extracted_targets = np.array(extracted_targets)
            for curr_cls in np.unique(extracted_targets):
                # get all indices from current class
                cls_ind = np.where(extracted_targets == curr_cls)[0]
                # get all extracted features for current class
                cls_feats = extracted_features[cls_ind]
                # add the exemplars to the set and normalize
                cls_feats_mean = cls_feats.mean(0) / cls_feats.mean(0).norm()
                self.exemplar_means.append(cls_feats_mean)

    # Algorithm 2: iCaRL Incremental Train
    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""

        # remove mean of exemplars during training since Alg. 1 is not used during Alg. 2
        self.exemplar_means = []

        # Algorithm 3: iCaRL Update Representation
        # Alg. 3. "form combined training set", add exemplars to train_loader
        if t > 0:
            trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory)

        # FINETUNING TRAINING -- contains the epochs loop
        super().train_loop(t, trn_loader, val_loader)

        # EXEMPLAR MANAGEMENT -- select training subset
        # Algorithm 4: iCaRL ConstructExemplarSet and Algorithm 5: iCaRL ReduceExemplarSet
        self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)

        # compute mean of exemplars
        self.compute_mean_of_exemplars(trn_loader, val_loader.dataset.transform)

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
        for images, targets in trn_loader:
            # Forward old model
            outputs_old = None
            if t > 0:
                outputs_old = self.model_old(images.to(self.device))
            # Forward current model
            outputs = self.model(images.to(self.device))
            loss = self.criterion(t, outputs, targets.to(self.device), outputs_old)
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

    def eval(self, t, val_loader):
        """Contains the evaluation code"""
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()
            for images, targets in val_loader:
                # Forward old model
                outputs_old = None
                if t > 0:
                    outputs_old = self.model_old(images.to(self.device))
                # Forward current model
                outputs, feats = self.model(images.to(self.device), return_features=True)
                loss = self.criterion(t, outputs, targets.to(self.device), outputs_old)
                # during training, the usual accuracy is computed on the outputs
                if not self.exemplar_means:
                    hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                else:
                    hits_taw, hits_tag = self.classify(t, feats, targets)
                # Log
                total_loss += loss.item() * len(targets)
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num += len(targets)
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    # Algorithm 3: classification and distillation terms -- original formulation has no trade-off parameter (lamb=1)
    def criterion(self, t, outputs, targets, outputs_old=None):
        """Returns the loss value"""

        # Classification loss for new classes
        loss = torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
        # Distillation loss for old classes
        if t > 0:
            # The original code does not match with the paper equation, maybe sigmoid could be removed from g
            g = torch.sigmoid(torch.cat(outputs[:t], dim=1))
            q_i = torch.sigmoid(torch.cat(outputs_old[:t], dim=1))
            loss += self.lamb * sum(torch.nn.functional.binary_cross_entropy(g[:, y], q_i[:, y]) for y in
                                    range(sum(self.model.task_cls[:t])))
        return loss

# +
# total_exem = [(1, 509), (1, 1121), (1, 44), (0, 146), (0, 651), (1, 1167), (1, 275), (0, 683), (0, 351), (1, 526), (0, 0), (1, 255), (1, 455), (1, 874), (0, 1071), (0, 1163), (1, 946), (1, 720), (0, 334), (0, 1186), (1, 829), (0, 1238), (1,1202), (1, 619), (0, 658), (0, 486), (0, 592), (0, 1082), (1, 569), (0, 1132), (0, 1061), (0, 420), (1, 1139), (0, 995), (0, 142), (1, 1208), (1, 196), (0, 437), (1, 1289), (0, 383), (0, 234), (1, 456), (0, 252), (0, 1046), (0, 364), (0, 831), (1, 1315), (1, 796), (1, 1338), (1, 883), (0, 919), (1, 517), (0, 674), (0, 240), (0, 294), (1, 400), (0, 128), (1, 259), (1, 446), (1, 586), (1, 909), (1, 350), (1, 114), (1, 1269), (0, 815), (1, 1306), (0, 967), (0, 869), (0, 751), (1, 342), (1,169), (0, 494), (1, 376), (0, 59), (0, 926), (0, 1131), (0, 251), (0, 1219), (0, 759), (0, 281), (0, 290), (0, 143), (1, 1241), (1, 1239), (1, 1231), (1, 1223), (1, 602), (0, 238), (1, 163), (1, 80), (1, 1005), (0, 1154), (0, 391), (0, 176), (1,286), (0, 1161), (1, 498), (0, 297), (1, 1136), (0, 217), (0, 945), (1, 43), (1, 270), (1, 574), (0, 1340), (1, 773), (0, 1177), (1, 646), (1, 1348), (0, 448), (0, 338), (0, 1233), (0, 42), (0, 1200), (0, 898), (1, 493), (1, 1268), (0, 389), (1,548), (1, 450), (0, 1302), (1, 670), (0, 403), (0, 101), (1, 1283), (0, 887), (1, 856), (0, 639), (0, 1309), (0, 162), (0,716), (1, 1065), (1, 969), (0, 972), (1, 142), (0, 915), (1, 795), (0, 961), (0, 838), (0, 1022), (0, 435), (0, 214), (1, 1048), (1, 772), (1, 398), (1, 463), (1, 214), (1, 651), (1, 1349), (0, 1317), (1, 976), (1, 611), (0, 186), (0, 154), (0, 150), (1, 1125), (0, 1062), (0, 966), (0, 1244), (1, 245), (0, 720), (1, 1227), (0, 978), (1, 537), (1, 482), (1, 352), (0,17), (1, 479), (1, 1168), (0, 397), (0, 48), (0, 712), (1, 488), (1, 361), (1, 252), (1, 676), (1, 219), (0, 512), (1, 1316), (0, 818), (1, 762), (1, 1229), (1, 1173), (1, 318), (0, 1246), (1, 1064), (1, 748), (0, 1315), (0, 428), (0, 1109), (0,399), (0, 1199), (0, 1183), (1, 1154), (1, 12), (1, 336), (0, 39), (0, 392), (1, 399), (1, 495)]

# +
# print(total_exem)

# +
# exem_0 = [total_exem[x][1] for x in range(len(total_exem)) if total_exem[x][0] == 0]
# exem_0
