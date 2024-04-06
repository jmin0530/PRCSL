# -*- coding: utf-8 -*-
import importlib
from argparse import ArgumentParser
import numpy as np

from datasets.memory_dataset import MemoryDataset


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
                          dp=False, prev_cls=None, fix_prev=False):
        if self._is_active():
            if t == 0:
                self.images, self.labels, self.exemplars_per_class = \
                self.exemplars_selector(server_model, client_model, client_loaders, selection_transform, dp, prev_cls, fix_prev)
                if self.exemplars_per_class != 0:
                    self.previous_labels = np.unique(self.labels).tolist()
                print('| Constructed {:d} exemplars'.format(len(self.images)))
            else:         
                new_exems_imgs, new_exems_lbls, self.exemplars_per_class = \
                self.exemplars_selector(server_model, client_model, client_loaders, selection_transform, dp, prev_cls, fix_prev)
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
