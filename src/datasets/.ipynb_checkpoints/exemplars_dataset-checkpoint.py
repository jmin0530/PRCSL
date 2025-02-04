import importlib
from argparse import ArgumentParser
import torch
from copy import deepcopy

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
        selector_cls = getattr(importlib.import_module(name='datasets.exemplars_selection'), cls_name)
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

    def collect_exemplars(self, model, trn_loader, selection_transform):
        if self._is_active():
            self.images, self.labels = self.exemplars_selector(model, trn_loader, selection_transform)
            
    def add_logits_to_memory(self, t, exemplars_dataset, model, exem_batch_size, device):
        '''In DER, memory logits shuold be loaded.'''
#         temp_exem_dataset = deepcopy(exemplars_dataset)
#         exem_loader = torch.utils.data.DataLoader(temp_exem_dataset, batch_size=exem_batch_size, shuffle=False, drop_last=False)
#         exem_loader.return_logits = False
        self.logits = []
        imgs = torch.stack(exemplars_dataset.images[:]).to(device)
        output = model(imgs)
        for i in range(len(output)):
            output[i] = output[i].clone().detach()
        output = torch.cat(output, dim=1)
        self.logits.extend(output.cpu())
        print("self.logits length and one shape: ", len(self.logits), self.logits[0].shape)

# +
# class ExtendedExemplarsDataset(ExemplarsDataset):
#     def __init__(self, transform, class_indices, num_exemplars=0, num_exemplars_per_class=0, exemplar_selection='random'):
#         super().__init__(transform, class_indices, num_exemplars, num_exemplars_per_class, exemplar_selection)
#         self.logits = None
#         self.task_labels = None
#         self.device = 'cpu'

#     def init_tensors(self, examples, labels, logits=None, task_labels=None):
#         self.examples = torch.zeros((self.max_num_exemplars, *examples.shape[1:]), dtype=torch.float32, device=self.device)
#         self.labels = torch.zeros((self.max_num_exemplars,), dtype=torch.int64, device=self.device)
#         if logits is not None:
#             self.logits = torch.zeros((self.max_num_exemplars, *logits.shape[1:]), dtype=torch.float32, device=self.device)
#         if task_labels is not None:
#             self.task_labels = torch.zeros((self.max_num_exemplars,), dtype=torch.int64, device=self.device)

#     def add_data(self, examples, labels, logits=None, task_labels=None):
#         for i in range(examples.shape[0]):
#             index = reservoir(self.num_seen_examples, self.max_num_exemplars)
#             self.num_seen_examples += 1
#             if index >= 0:
#                 self.examples[index] = examples[i].to(self.device)
#                 self.labels[index] = labels[i].to(self.device)
#                 if logits is not None:
#                     self.logits[index] = logits[i].to(self.device)
#                 if task_labels is not None:
#                     self.task_labels[index] = task_labels[i].to(self.device)

#     def get_data(self, size, transform=None, device=None):
#         indices = np.random.choice(len(self.examples), size=size, replace=False)
#         selected_examples = self.examples[indices].to(device or self.device)
#         selected_labels = self.labels[indices].to(device or self.device)
#         if transform is not None:
#             selected_examples = apply_transform(selected_examples, transform=transform)
#         return selected_examples, selected_labels
