from os.path import join
import os

_BASE_DATA_PATH = "./data"

# +
dataset_config = {
    'mnist': {
        'path': join(_BASE_DATA_PATH, 'mnist'),
        'normalize': ((0.1307,), (0.3081,)),
        # Use the next 3 lines to use MNIST with a 3x32x32 input
        # 'extend_channel': 3,
        # 'pad': 2,
        # 'normalize': ((0.1,), (0.2752,))    # values including padding
    },
    'svhn': {
        'path': join(_BASE_DATA_PATH, 'svhn'),
        'resize': None,
        'crop': None,
        'flip': False,
        'vertical_flip': False,
        'rotation': None,
        'normalize': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        'gaussian': False
    },
    'pathmnist': {
        'path': join(_BASE_DATA_PATH, 'pathmnist'),
        'resize': None,
        'pad': 4,
        'crop': 32,
        'flip': True,
        'vertical_flip': False,
        'rotation': None,
        'normalize': None,
        'gaussian': False
    },
    'organamnist': {
        'path': join(_BASE_DATA_PATH, 'organamnist'),
        'resize': None,
        'pad': 4,
        'crop': 32,
        'flip': True,
        'vertical_flip': False,
        'rotation': None,
        'normalize': None,
        'gaussian': False
    },
    'bloodmnist': {
        'path': join(_BASE_DATA_PATH, 'bloodmnist'),
        'resize': None,
        'pad': 4,
        'crop': 32,
        'flip': True,
        'vertical_flip': False,
        'rotation': None,
        'normalize': None,
        'class_order': [1,6,3,7,2,5,0,4]
    },
    'dermamnist': {
        'path': join(_BASE_DATA_PATH, 'dermamnist'),
        'resize': None,
        'pad': 4,
        'crop': 32,
        'flip': True,
        'vertical_flip': False,
        'rotation': None,
        'normalize': None,
        'class_order': [
            0,1,2,4,5,6,3
        ],
        'gaussian': False
    },
    'tissuemnist': {
        'path': join(_BASE_DATA_PATH, 'tissuemnist'),
        'resize': None,
        'pad': 4,
        'crop': 32,
        'flip': True,
        'vertical_flip': False,
        'rotation': None,
        'normalize': None,
        'gaussian': False
    },
    'cifar100_icarl': { 
        'path': join(_BASE_DATA_PATH, 'cifar100'),
        'resize': None,
        'pad': 4,
        'crop': 32,
        'flip': True,
        'vertical_flip': False,
        'rotation': None,
        'normalize': ((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)), 
        'class_order': [
            68, 56, 78, 8, 23, 84, 90, 65, 74, 76, 40, 89, 3, 92, 55, 9, 26, 80, 43, 38, 58, 70, 77, 1, 85, 19, 17, 50,
            28, 53, 13, 81, 45, 82, 6, 59, 83, 16, 15, 44, 91, 41, 72, 60, 79, 52, 20, 10, 31, 54, 37, 95, 14, 71, 96,
            98, 97, 2, 64, 66, 42, 22, 35, 86, 24, 34, 87, 21, 99, 0, 88, 27, 18, 94, 11, 12, 47, 25, 30, 46, 62, 69,
            36, 61, 7, 63, 75, 5, 32, 4, 51, 48, 73, 93, 39, 67, 29, 49, 57, 33
        ], 
    },
    'ham10000': {
        'path': join(_BASE_DATA_PATH, 'ham10000'),
        'resize': (224, 224),
        'vertical_flip': True,
        'normalize': ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        'rotation': False,
        'class_order': [5,4,2,1,0,6,3],
        'gaussian': False
    },
    'cch5000': {
        'path': join(_BASE_DATA_PATH, 'cch5000'),
        'resize': (256, 256),
        'crop': (224, 224),
        'flip': True,
        'vertical_flip': False,
        'normalize': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        'rotation': False,
        'gaussian': False,
        'class_order': [2, 3, 0, 4, 5, 6, 7, 1]
    }
    
}

# Add missing keys:
for dset in dataset_config.keys():
    for k in ['resize', 'pad', 'crop', 'normalize', 'class_order', 'extend_channel']:
        if k not in dataset_config[dset].keys():
            dataset_config[dset][k] = None
    if 'flip' not in dataset_config[dset].keys():
        dataset_config[dset]['flip'] = False
