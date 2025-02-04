import os
import torch
import random
import numpy as np

cudnn_deterministic = True


def seed_everything(seed=0):
    """Fix all random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = False 


def print_summary(acc_taw, acc_tag, forg_taw, forg_tag):
    """Print summary of results"""
    for name, metric in zip(['TAw Acc', 'TAg Acc', 'TAw Forg', 'TAg Forg'], [acc_taw, acc_tag, forg_taw, forg_tag]):
        print('*' * 108)
        print(name)
        print_horiz = []
        for i in range(metric.shape[0]):
            print('\t', end='')
            for j in range(metric.shape[1]):
                print('{:5.2f}% '.format(100 * metric[i, j]), end='')
            if np.trace(metric) == 0.0:
                if i > 0:
                    print('\tAvg.:{:5.2f}% '.format(100 * metric[i, :i].mean()), end='')
                    print_horiz.append(100 * metric[i, :i].mean())
            else:
                print('\tAvg.:{:5.2f}% '.format(100 * metric[i, :i + 1].mean()), end='')
                print_horiz.append(100 * metric[i, :i + 1].mean())
            print()
            
        print(name + ' ' + 'result')
        for r in print_horiz:
            print('{:5.2f}'.format(r), end='\t')
        print()
    print('*' * 108)
