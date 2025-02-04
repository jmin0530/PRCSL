# -*- coding: utf-8 -*-
import random
import time
from contextlib import contextmanager
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.transforms import Lambda
from copy import deepcopy

from datasets.exemplars_dataset_split import ExemplarsDataset
from networks.network import LLL_Net


class ExemplarsSelector:
    """Exemplar selector for approaches with an interface of Dataset"""

    def __init__(self, exemplars_dataset: ExemplarsDataset):
        self.exemplars_dataset = exemplars_dataset

    def __call__(self, server_model, client_model, client_loaders, transform, dp, prev_cls, fix_prev, t, taskcla=None):
        clock0 = time.time()
        exemplars_per_class = self._exemplars_per_class_num(server_model, t, taskcla)
        total_selected_indices = []
        features = []
        targets = []
        c_client_loaders = deepcopy(client_loaders)
        for i in range(len(c_client_loaders)):
            # client dataset selection 
            with override_dataset_transform(c_client_loaders[i][0].dataset, transform) as ds_for_selection:
                # change loader and fix to go sequentially (shuffle=False), keeps same order for later, eval transforms
                sel_loader = DataLoader(ds_for_selection, batch_size=c_client_loaders[i][0].batch_size, shuffle=False,
                                        num_workers=c_client_loaders[i][0].num_workers, pin_memory=c_client_loaders[i][0].pin_memory)
                client_features, client_targets = self._extract_feats(server_model, client_model, sel_loader, transform)
                features.append(client_features)
                targets.extend(client_targets.tolist())
                if i == 0:
                    # first client data nums
                    c1_data_nums = client_features.shape[0]  
        extracted_features = torch.cat(features, 0)
        last_client_start_ind = (len(c_client_loaders)-1) * c1_data_nums
        assert c1_data_nums == features[0].shape[0], "client 1 data num is wrong"
        targets = np.array(targets)
        
        selected_info = self._select_indices(extracted_features, targets, \
                                            exemplars_per_class, c1_data_nums, last_client_start_ind, \
                                            dp, prev_cls, fix_prev)



        new_exems_imgs = []
        new_exems_lbls = []

        for _ in selected_info:
            client_idx, data_idx = _[0], _[1]  
            if client_idx >= len(c_client_loaders):
                continue
            with override_dataset_transform(c_client_loaders[client_idx][0].dataset, Lambda(lambda x: np.array(x))) as ds_for_raw:
                x, y, _ = zip(*(ds_for_raw[idx] for idx in [data_idx]))
            new_exems_imgs.append(x[0])
            new_exems_lbls.append(y[0])

        clock1 = time.time()
        print('| Selected {:d} train exemplars, time={:5.1f}s'.format(len(new_exems_imgs), clock1 - clock0))
        return new_exems_imgs, new_exems_lbls, exemplars_per_class

    def _exemplars_per_class_num(self, model: LLL_Net, t, taskcla=None):
        if self.exemplars_dataset.max_num_exemplars_per_class:
            return self.exemplars_dataset.max_num_exemplars_per_class
        
        try:
            num_cls = model.task_cls.sum().item()
        except:
            num_cls = sum([n[1] for n in taskcla[:t+1]])
        num_exemplars = self.exemplars_dataset.max_num_exemplars
        exemplars_per_class = int(np.ceil(num_exemplars / num_cls))
        assert exemplars_per_class > 0, \
            "Not enough exemplars to cover all classes!\n" \
            "Number of classes so far: {}. " \
            "Limit of exemplars: {}".format(num_cls,
                                            num_exemplars)
        return exemplars_per_class

    def _select_indices(self, model: LLL_Net, sel_loader: DataLoader, exemplars_per_class: int, transform) -> Iterable:
        pass


class RandomExemplarsSelector(ExemplarsSelector):
    """Selection of new samples. This is based on random selection, which produces a random list of samples."""

    def __init__(self, exemplars_dataset):
        super().__init__(exemplars_dataset)

    def _select_indices(self, model: LLL_Net, sel_loader: DataLoader, exemplars_per_class: int, transform, dp, prev_cls, fix_prev) -> Iterable:
        num_cls = sum(model.task_cls)
        result = []
        labels = self._get_labels(sel_loader)
        for curr_cls in range(num_cls):
            if prev_cls is not None and (curr_cls in prev_cls) and fix_prev: # when apply dp to exemplar datas, do not select previous task classes again
                continue
            # get all indices from current class -- check if there are exemplars from previous task in the loader
            cls_ind = np.where(labels == curr_cls)[0]
            assert (len(cls_ind) > 0), "No samples to choose from for class {:d}".format(curr_cls)
            assert (exemplars_per_class <= len(cls_ind)), "Not enough samples to store"
            # select the exemplars randomly
            result.extend(random.sample(list(cls_ind), exemplars_per_class))
        return result

    def _get_labels(self, sel_loader):
        if hasattr(sel_loader.dataset, 'labels'):  # BaseDataset, MemoryDataset
            labels = np.asarray(sel_loader.dataset.labels)
        elif isinstance(sel_loader.dataset, ConcatDataset):
            labels = []
            for ds in sel_loader.dataset.datasets:
                labels.extend(ds.labels)
            labels = np.array(labels)
        else:
            raise RuntimeError("Unsupported dataset: {}".format(sel_loader.dataset.__class__.__name__))
        return labels


class HerdingExemplarsSelector(ExemplarsSelector):
    """Selection of new samples. This is based on herding selection, which produces a sorted list of samples of one
    class based on the distance to the mean sample of that class. From iCaRL algorithm 4 and 5:
    https://openaccess.thecvf.com/content_cvpr_2017/papers/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.pdf
    """
    def __init__(self, exemplars_dataset):
        super().__init__(exemplars_dataset)

    def _extract_feats(self, server_model, client_model, sel_loader: DataLoader, transform) -> Iterable:
        model_device = next(server_model.parameters()).device  # we assume here that whole model is on a single device

        # extract outputs from the model for all train samples
        extracted_features = []
        extracted_targets = []
        with torch.no_grad():
            server_model.eval()
            client_model.eval()
            for images, targets, _ in sel_loader:
                client_feats = client_model(images.to(model_device))
                feats = server_model(client_feats, return_features=True)[1]
                feats = feats / (feats.norm(dim=1).view(-1, 1) + 1e-9)  # Feature normalization
                extracted_features.append(feats)
                extracted_targets.extend(targets)
        extracted_features = (torch.cat(extracted_features)).cpu()
        extracted_targets = np.array(extracted_targets)
        
        return extracted_features, extracted_targets
        
    def _select_indices(self, extracted_features, extracted_targets, exemplars_per_class, c1_data_nums, 
                        last_client_start_ind, dp, prev_cls, fix_prev):
        selected_info = []
        print("-"*50)
        # iterate through all classes
        for curr_cls in np.unique(extracted_targets):
            # get all indices from current class
            cls_ind = np.where(extracted_targets == curr_cls)[0]
            assert (len(cls_ind) > 0), "No samples to choose from for class {:d}".format(curr_cls)
            assert (exemplars_per_class <= len(cls_ind)), "Not enough samples to store"
            # get all extracted features for current class
            cls_feats = extracted_features[cls_ind]
            # calculate the mean
            cls_mu = cls_feats.mean(0)
            # select the exemplars closer to the mean of each class
            selected = []
            selected_feat = []
            for k in range(exemplars_per_class):
                # fix this to the dimension of the model features
                sum_others = torch.zeros(cls_feats.shape[1])
                for j in selected_feat:
                    sum_others += j / (k + 1)
                dist_min = np.inf
                # choose the closest to the mean of the current class
                for item in cls_ind:
                    if item not in selected:
                        feat = extracted_features[item]
                        dist = torch.norm(cls_mu - feat / (k + 1) - sum_others)
                        if dist < dist_min:
                            dist_min = dist
                            newone = item
                            newonefeat = feat
                selected_feat.append(newonefeat) # save selected features
                selected.append(newone) # save selected features index
                
                if last_client_start_ind == 0: # when client per task number is one
                    selected_info.append([newone // c1_data_nums, newone % c1_data_nums])
                else:
                    if newone >= last_client_start_ind:
                        selected_info.append([newone // c1_data_nums, newone % last_client_start_ind])
                    else:
                        selected_info.append([newone // c1_data_nums, newone % c1_data_nums])
        print("-"*50)
        return selected_info


def dataset_transforms(dataset, transform_to_change):
    if isinstance(dataset, ConcatDataset):
        r = []
        for ds in dataset.datasets:
            r += dataset_transforms(ds, transform_to_change)
        return r
    else:
        old_transform = dataset.transform
        dataset.transform = transform_to_change
        return [(dataset, old_transform)]


@contextmanager
def override_dataset_transform(dataset, transform):
#     datasets_with_orig_transform = dataset_transforms(dataset, transform)
    try:
        datasets_with_orig_transform = dataset_transforms(dataset, transform)
        yield dataset
    finally:
        # get bac original transformations
        for ds, orig_transform in datasets_with_orig_transform:
            ds.transform = orig_transform
