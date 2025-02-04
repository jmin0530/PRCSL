# -*- coding: utf-8 -*-
import importlib
from argparse import ArgumentParser
import numpy as np
import torch

from datasets.memory_dataset import MemoryDataset

from copy import deepcopy
# from contextlib import contextmanager

class ExemplarsDataset(MemoryDataset):
    """Exemplar storage for approaches with an interface of Dataset"""

    def __init__(self, transform, class_indices,
                 num_exemplars=0, num_exemplars_per_class=0, exemplar_selection='random'):
        super().__init__({'x': [], 'y': []}, transform, class_indices=class_indices)
        self.max_num_exemplars_per_class = num_exemplars_per_class
        self.max_num_exemplars = num_exemplars
        assert (num_exemplars_per_class == 0) or (num_exemplars == 0), 'Cannot use both limits at once!'
        cls_name = "{}ExemplarsSelector".format(exemplar_selection.capitalize())
        selector_cls = getattr(importlib.import_module(name='datasets.exemplars_selection_split'), cls_name)
        self.exemplars_selector = selector_cls(self)

    # Returns a parser containing the approach specific parameters
    @staticmethod
    def extra_parser(args):
        parser = ArgumentParser("Exemplars Management Parameters")
        _group = parser.add_mutually_exclusive_group()
        _group.add_argument('--num-exemplars', default=0, type=int, required=False,
                            help='Fixed memory, total number of exemplars (default=%(default)s)')
        _group.add_argument('--num-exemplars-per-class', default=0, type=int, required=False,
                            help='Growing memory, number of exemplars per class (default=%(default)s)')
        parser.add_argument('--exemplar-selection', default='random', type=str,
                            choices=['herding', 'random', 'entropy', 'distance'],
                            required=False, help='Exemplar selection strategy (default=%(default)s)')
        return parser.parse_known_args(args)

    def _is_active(self):
        return self.max_num_exemplars_per_class > 0 or self.max_num_exemplars > 0

    def collect_exemplars(self, t, server_model, client_model, task_offset, client_loaders, selection_transform, \
                          dp=False, prev_cls=None, fix_prev=False, taskcla=None):
        if self._is_active():
            if t == 0:
                self.images, self.labels, self.exemplars_per_class = \
                self.exemplars_selector(server_model, client_model, client_loaders, selection_transform, dp, prev_cls, fix_prev, \
                                       t, taskcla)
                if self.exemplars_per_class != 0:
                    self.previous_labels = np.unique(self.labels).tolist()
                print('| Constructed {:d} exemplars'.format(len(self.images)))
            else:         
                new_exems_imgs, new_exems_lbls, self.exemplars_per_class = \
                self.exemplars_selector(server_model, client_model, client_loaders, selection_transform, dp, prev_cls, fix_prev, t, taskcla)
                if not dp:
                    for prev_cls in range(len(self.previous_labels)):
                        prev_cls_ind = np.where(np.array(list(self.labels))==prev_cls)[0]
                        del self.images[prev_cls_ind[self.exemplars_per_class]:prev_cls_ind[-1]+1]
                        del self.labels[prev_cls_ind[self.exemplars_per_class]:prev_cls_ind[-1]+1]
                    self.images += new_exems_imgs
                    self.labels += new_exems_lbls
                else:
                    self.images.extend(new_exems_imgs)
                    self.labels.extend(new_exems_lbls)
                
                if self.exemplars_per_class != 0:
                    self.previous_labels = list(np.unique(self.labels))
                print('| Constructed {:d} exemplars'.format(len(self.images)))
                
    def add_logits_to_memory(self, t, exemplars_dataset, client_loaders, client_model, server_model, exem_batch_size, device):
        '''In DER, memory logits shuold be loaded instead of labels'''
        temp_exem_dataset = deepcopy(exemplars_dataset)
        exem_loader = torch.utils.data.DataLoader(temp_exem_dataset, batch_size=exem_batch_size, shuffle=False, drop_last=False)
        exem_loader.return_logits = False
        self.logits = []
        for data in exem_loader:
#             print("data[0] shape: ", data[0].shape)
#             print("data[1] shape: ", data[1].shape)
            img = data[0]
            client_feats = client_model(img.to(device))
            output = server_model(client_feats)
            for i in range(len(output)):
                output[i] = output[i].clone().detach()
#             memory_logit_outputs = output.clone().detach().squeeze(0)
#             print("memory_output_logit shape: ", memory_output_logit.shape)
#             print("memory_output_logit type: ", type(memory_output_logit))
#             logit_outputs = memory_output_logit[0].clone().detach()
#             print("logit_outputs: ", logit_outputs.shape)
#             if t != 0:
            output = torch.cat(output, dim=1)
#             print("memory logit shape: ", output.shape)
            self.logits.extend(output.cpu().numpy())
#             else:
#                 logits.append(output)
#         logits = torch.cat(logits, 0)
#         print("memory logits shape: ", logits.shape)
        print("self.logits length and one shape: ", len(self.logits), self.logits[0].shape)
#         return [logits]


# +
# class ExtendedExemplarsDataset(ExemplarsDataset):
#     """Extended Exemplar storage to include logits and task labels for split learning and continual learning."""

#     def __init__(self, transform, class_indices, num_exemplars=0, num_exemplars_per_class=0, exemplar_selection='random'):
#         super().__init__(transform, class_indices, num_exemplars, num_exemplars_per_class, exemplar_selection)
#         self.logits = None
#         self.task_labels = None
#         self.device = 'cpu'
#         self.num_seen_examples = 0
#         self.attributes = ['examples', 'labels', 'logits', 'task_labels']

#     def init_tensors(self, examples, labels, logits=None, task_labels=None):
#         """Initializes required tensors based on the provided examples."""
#         super().init_tensors(examples, labels)  # Call to the parent class init_tensors
#         if logits is not None:
#             self.logits = torch.zeros((self.max_num_exemplars, *logits.shape[1:]), dtype=torch.float32, device=self.device)
#         if task_labels is not None:
#             self.task_labels = torch.zeros((self.max_num_exemplars,), dtype=torch.int64, device=self.device)

#     def add_data(self, examples, labels, logits=None, task_labels=None):
#         """Adds the data to the memory buffer according to a reservoir sampling strategy."""
#         if not hasattr(self, 'examples'):
#             self.init_tensors(examples, labels, logits, task_labels)

#         for i in range(examples.shape[0]):
#             index = self.reservoir(self.num_seen_examples, self.max_num_exemplars)
#             self.num_seen_examples += 1
#             if index >= 0:
#                 self.examples[index] = examples[i].to(self.device)
#                 self.labels[index] = labels[i].to(self.device)
#                 if logits is not None:
#                     self.logits[index] = logits[i].to(self.device)
#                 if task_labels is not None:
#                     self.task_labels[index] = task_labels[i].to(self.device)

#     def get_data(self, size, transform=None, device=None, return_index=False):
#         """Randomly samples a batch of 'size' items from the buffer."""
#         target_device = self.device if device is None else device
#         indices = np.random.choice(min(self.num_seen_examples, self.max_num_exemplars), size=size, replace=False)

#         selected_examples = self.examples[indices].to(target_device)
#         selected_labels = self.labels[indices].to(target_device)
#         if transform is not None:
#             selected_examples = apply_transform(selected_examples, transform=transform)

#         result = (selected_examples, selected_labels)
#         if return_index:
#             result = (indices,) + result
#         if self.logits is not None:
#             result += (self.logits[indices].to(target_device),)
#         if self.task_labels is not None:
#             result += (self.task_labels[indices].to(target_device),)

#         return result

#     def reservoir(self, num_seen_examples, buffer_size):
#         """Reservoir sampling algorithm."""
#         if num_seen_examples < buffer_size:
#             return num_seen_examples
#         rand = np.random.randint(0, num_seen_examples + 1)
#         return rand if rand < buffer_size else -1

#     def empty(self):
#         """Empties the exemplars in the memory buffer."""
#         self.num_seen_examples = 0
#         self.examples.fill_(0)
#         self.labels.fill_(0)
#         if self.logits is not None:
#             self.logits.fill_(0)
#         if self.task_labels is not None:
#             self.task_labels.fill_(0)

#     def is_empty(self):
#         """Checks if the buffer is empty."""
#         return self.num_seen_examples == 0

#     def get_all_data(self, transform=None, device=None):
#         """Returns all items currently in the memory buffer."""
#         target_device = self.device if device is None else device
#         data = (self.examples[:self.num_seen_examples].to(target_device),
#                 self.labels[:self.num_seen_examples].to(target_device))
#         if transform is not None:
#             data = (apply_transform(data[0], transform=transform), data[1])
#         if self.logits is not None:
#             data += (self.logits[:self.num_seen_examples].to(target_device),)
#         if self.task_labels is not None:
#             data += (self.task_labels[:self.num_seen_examples].to(target_device),)
#         return data
