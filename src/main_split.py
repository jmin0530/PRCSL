# -*- coding: utf-8 -*-
from enum import unique
import os
import time
import torch
import argparse
import importlib
import numpy as np
from functools import reduce
import torch.nn.functional as F
import torch.nn as nn
import math
from copy import deepcopy

import utils
import approach
from loggers.exp_logger import MultiLogger
from datasets.data_loader_split import get_final_datas, make_client_loaders
from datasets.dataset_config import dataset_config
from networks import tvmodels, allmodels, set_tvmodel_head_var
from networks.resnet18_split import *
from networks.resnet32_split import *
import datasets.memory_dataset as memd
import datasets.base_dataset as basedat
from networks.network import LLL_Net

import pdb


# +
def main(argv=None):
    tstart = time.time()
    # Arguments
    parser = argparse.ArgumentParser(description='Alleviating Catastrophic Forgetting with Privacy Preserving Distributed Learning')

    # miscellaneous args
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU (default=%(default)s)')
    parser.add_argument('--results-path', type=str, default='../results',
                        help='Results path (default=%(default)s)')
    parser.add_argument('--exp-name', default=None, type=str,
                        help='Experiment name (default=%(default)s)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed (default=%(default)s)')
    parser.add_argument('--log', default=['disk'], type=str, choices=['disk', 'tensorboard'],
                        help='Loggers used (disk, tensorboard) (default=%(default)s)', nargs='*', metavar="LOGGER")
    parser.add_argument('--save-models', action='store_true',
                        help='Save trained models (default=%(default)s)')
    parser.add_argument('--no-cudnn-deterministic', action='store_true',
                        help='Disable CUDNN deterministic (default=%(default)s)')
    # dataset args
    parser.add_argument('--datasets', default=['cifar100'], type=str, choices=list(dataset_config.keys()),
                        help='Dataset or datasets used (default=%(default)s)', nargs='+', metavar="DATASET")
    parser.add_argument('--exem-batch-size', default=128, type=int, required=False,
                        help='Exemplar dataloader batch_size (default=%(default)s)')
    parser.add_argument('--num-workers', default=2, type=int, required=False,
                        help='Number of subprocesses to use for dataloader (default=%(default)s)')
    parser.add_argument('--pin-memory', default=False, type=bool, required=False,
                        help='Copy Tensors into CUDA pinned memory before returning them (default=%(default)s)')
    parser.add_argument('--batch-size', default=64, type=int, required=False,
                        help='Number of samples per batch to load (default=%(default)s)')
    parser.add_argument('--num-tasks', default=4, type=int, required=False,
                        help='Number of tasks per dataset (default=%(default)s)')
    parser.add_argument('--nc-first-task', default=None, type=int, required=False,
                        help='Number of classes of the first task (default=%(default)s)')
    parser.add_argument('--use-valid-only', action='store_true',
                        help='Use validation split instead of test (default=%(default)s)')
    parser.add_argument('--stop-at-task', default=0, type=int, required=False,
                        help='Stop training after specified task (default=%(default)s)')
    # model args
    parser.add_argument('--network', default='resnet32', type=str, choices=allmodels,
                        help='Network architecture used (default=%(default)s)', metavar="NETWORK")
    parser.add_argument('--keep-existing-head', action='store_true',
                        help='Disable removing classifier last layer (default=%(default)s)')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained backbone (default=%(default)s)')
    # training args
    parser.add_argument('--approach', default='finetuning', type=str, choices=approach.__all__,
                        help='Learning approach used (default=%(default)s)', metavar="APPROACH")
    parser.add_argument('--dp-mean-batch', default=3, type=int, required=False,
                        help='DP mean batch')
    parser.add_argument('--epsilon', default=10.0, type=float, required=False,
                        help='DP epsilon')
    parser.add_argument('--alpha', default=0.03, type=float, required=False,
                        help='DER approach regularization coefficient')
    parser.add_argument('--eval-on-train', action='store_true',
                        help='Show train loss and accuracy (default=%(default)s)')
    parser.add_argument('--fix-bn', action='store_true',
                        help='Fix batch normalization after first task (default=%(default)s)')
    parser.add_argument('--nepochs', default=100, type=int, required=False,
                        help='Number of epochs per training session (default=%(default)s)')
    parser.add_argument('--nclients', default=10, type=int, required=False,
                        help='Number of clients per one task (default=%(default)s)')
    parser.add_argument('--opt', default='adam', type=str,
                        help='Choose optimizer')
    parser.add_argument('--lamb-distill', default=1.0, type=float, required=False,
                        help='Distillation loss lambda')
    parser.add_argument('--lamb-distill-ewc', default=5000, type=float, required=False,
                        help='EWC penalty loss coeff')
    parser.add_argument('--lamb-distill-mas', default=1.0, type=float, required=False,
                        help='MAS penalty loss coeff')
    parser.add_argument('--lr', default=0.001, type=float, required=False,
                        help='Starting learning rate (default=%(default)s)')
    parser.add_argument('--lr-factor', default=10, type=float, required=False,
                        help='Learning rate decreasing factor (default=%(default)s)')
    parser.add_argument('--lr-finetune-factor', default=0.1, type=float, required=False,
                        help='Robust training learning rate factor (default=%(default)s)')
    parser.add_argument('--lr-patience', default=5, type=int, required=False,
                        help='Maximum patience to wait before decreasing learning rate (default=%(default)s)')
    parser.add_argument('--clipping', default=10000, type=float, required=False,
                        help='Clip gradient norm (default=%(default)s)')
    parser.add_argument('--momentum', default=0.0, type=float, required=False,
                        help='Momentum factor (default=%(default)s)')
    parser.add_argument('--weight-decay', default=0.0, type=float, required=False,
                        help='Weight decay (L2 penalty) (default=%(default)s)')
    parser.add_argument('--multi-softmax', action='store_true',
                        help='Apply separate soffix_prevtmax for each task (default=%(default)s)')
    parser.add_argument('--fix-prev', action='store_true',
                        help='Fix previous class exemplar feature means')
    parser.add_argument('--keep-classifier', action='store_true', help='Keep model\'s classifier')

    # Args
    args, extra_args = parser.parse_known_args(argv)
    args.results_path = os.path.expanduser(args.results_path)
    base_kwargs = dict(nepochs=args.nepochs, lr=args.lr, lr_factor=args.lr_factor,
                       lr_patience=args.lr_patience, clipgrad=args.clipping, momentum=args.momentum,
                       wd=args.weight_decay, multi_softmax=args.multi_softmax, fix_bn=args.fix_bn, eval_on_train=args.eval_on_train, 
                       exem_batch_size=args.exem_batch_size)

    if args.no_cudnn_deterministic:
        print('WARNING: CUDNN Deterministic will be disabled.')
        utils.cudnn_deterministic = False

    utils.seed_everything(seed=args.seed)
    print('=' * 108)
    print('Arguments =')
    for arg in np.sort(list(vars(args).keys())):
        print('\t' + arg + ':', getattr(args, arg))
    print('=' * 108)

    # Args -- CUDA
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = 'cuda'
    else:
        print('WARNING: [CUDA unavailable] Using CPU instead!')
        device = 'cpu'
    
    if len(args.datasets)==1 and (args.datasets[-1]=='tissuemnist' or args.datasets[-1]=='organamnist'):
        input_channel = 1
    else:
        input_channel = 3
        
    # Model initialization
    
    num_classes_dict = {'bloodmnist': 8, 'organamnist': 11, 'pathmnist': 9, 'tissuemnist': 8, 'cch5000': 8, 'ham10000': 7,
                       'cifar100_icarl': 100, 'svhn': 10}
    
    if args.datasets == ['cch5000'] or args.datasets == ['ham10000']:
        print("Network: ResNet18_split")
        init_server_model = ResNet18_server_side(Basicblock_resnet18, [2,2,2]).to(device)
        global_server_net = LLL_Net(init_server_model, remove_existing_head=not args.keep_existing_head, \
                                    keep_classifier=args.keep_classifier, num_classes=num_classes_dict[args.datasets[0]])
        global_client_net = ResNet18_client_side(Basicblock_resnet18, input_channel=input_channel).to(device)
        
    else:
        print("Network: ResNet32_split")
        init_server_model = ResNet32_server_side(BasicBlock, [5,5]).to(device)
        global_server_net = LLL_Net(init_server_model, remove_existing_head=not args.keep_existing_head, \
                                    keep_classifier=args.keep_classifier, num_classes=num_classes_dict[args.datasets[0]])
        global_client_net = ResNet32_client_side(BasicBlock, [5], input_channel).to(device)
    

    # Args -- Continual Learning Approach
    if args.approach == 'der_split_dp':
        from approach.incremental_learning_split_der import Inc_Learning_Appr
    else:
        from approach.incremental_learning_split import Inc_Learning_Appr
    Appr = getattr(importlib.import_module(name='approach.' + args.approach), 'Appr')
    assert issubclass(Appr, Inc_Learning_Appr)
    appr_args, extra_args = Appr.extra_parser(extra_args)
    print('Approach arguments =')
    for arg in np.sort(list(vars(appr_args).keys())):
        print('\t' + arg + ':', getattr(appr_args, arg))
    print('=' * 108)

    # Args -- Exemplars Management
    from datasets.exemplars_dataset_split import ExemplarsDataset
    Appr_ExemplarsDataset = Appr.exemplars_dataset_class()
    if Appr_ExemplarsDataset:
        assert issubclass(Appr_ExemplarsDataset, ExemplarsDataset)
        appr_exemplars_dataset_args, extra_args = Appr_ExemplarsDataset.extra_parser(extra_args)
        print('Exemplars dataset arguments =')
        for arg in np.sort(list(vars(appr_exemplars_dataset_args).keys())):
            print('\t' + arg + ':', getattr(appr_exemplars_dataset_args, arg))
        print('=' * 108)
    else:
        appr_exemplars_dataset_args = argparse.Namespace()

    assert len(extra_args) == 0, "Unused args: {}".format(' '.join(extra_args))

    # Log all arguments
    full_exp_name = reduce((lambda x, y: x[0] + y[0]), args.datasets) if len(args.datasets) > 0 else args.datasets[0]
    full_exp_name += '_' + args.approach
    if args.exp_name is not None:
        full_exp_name += '_' + args.exp_name
    logger = MultiLogger(args.results_path, full_exp_name, loggers=args.log, save_models=args.save_models)
    logger.log_args(argparse.Namespace(**args.__dict__, **appr_args.__dict__, **appr_exemplars_dataset_args.__dict__))

    # Load dataset
    utils.seed_everything(seed=args.seed)
    trn_data, val_data, tst_data, taskcla, class_indices, trn_transform, tst_transform \
    = get_final_datas(args.datasets, args.num_tasks, args.nc_first_task, \
                  args.batch_size, num_workers=args.num_workers, \
                  pin_memory=args.pin_memory)

    max_task = len(taskcla) if args.stop_at_task == 0 else args.stop_at_task

    utils.seed_everything(seed=args.seed)

    appr_kwargs = {**base_kwargs, **dict(logger=logger, **appr_args.__dict__)}
    if Appr_ExemplarsDataset:
        appr_kwargs['exemplars_dataset'] = Appr_ExemplarsDataset(trn_transform, class_indices,
                                                                 **appr_exemplars_dataset_args.__dict__)
    utils.seed_everything(seed=args.seed)
    appr = Appr(global_server_net, global_client_net, device, **appr_kwargs)

    # Loop tasks
    print("Taskcla: ", taskcla)
    acc_taw = np.zeros((max_task, max_task))
    acc_tag = np.zeros((max_task, max_task))
    forg_taw = np.zeros((max_task, max_task))
    forg_tag = np.zeros((max_task, max_task))
    
    appr.taskcla = taskcla
    
    # Train arguments
    appr.opt = args.opt
    appr.lamb = args.lamb_distill
    appr.ewc_lamb = args.lamb_distill_ewc
    appr.mas_lamb = args.lamb_distill_mas
    appr.exem_per_class = getattr(appr_exemplars_dataset_args, "num_exemplars_per_class")
    if args.approach == 'der_split_dp':
         appr.alpha = args.alpha
    
    # DP hyperparameters
    appr.epsilon = args.epsilon
    appr.dp_mean_batch = args.dp_mean_batch
    appr.fix_prev = args.fix_prev
    appr.lr_finetune_factor = args.lr_finetune_factor
    
    # Number of clients per task
    num_clients = [args.nclients for _ in range(args.num_tasks)]
    assert len(num_clients) == args.num_tasks, 'args.num_tasks should equal to len(num_clients)!!'
    print("Number of clients each task: ", num_clients)
    
    previous_client_loaders = []
    
    for t, (_, ncla) in enumerate(taskcla):
        current_num_client = num_clients[t] 
        current_task_trn_dataset = trn_data[t]
        current_task_val_dataset = val_data[t]
        current_task_tst_dataset = tst_data[t]
        client_data = {}
        for c in range(num_clients[t]):
            client_data[c] = {}
            client_data[c]['trn'] = {'x': [], 'y': []}
            client_data[c]['val'] = {'x': [], 'y': []}
            client_data[c]['tst'] = {'x': [], 'y': []}
        
        trn_dataset_labels = torch.tensor(trn_data[t]['y'])
        val_dataset_labels = torch.tensor(val_data[t]['y'])
        tst_dataset_labels = torch.tensor(tst_data[t]['y'])
        
        trn_unique_lbls = torch.unique(trn_dataset_labels)
        val_unique_lbls = torch.unique(val_dataset_labels)
        test_unique_lbls = torch.unique(tst_dataset_labels)
        
        trn_frequencies = torch.Tensor([len(torch.where(trn_dataset_labels==lbl)[0]) for lbl in trn_unique_lbls]).to(device).float()
        val_frequencies = torch.Tensor([len(torch.where(val_dataset_labels==lbl)[0]) for lbl in val_unique_lbls]).to(device).float()
        test_frequencies = torch.Tensor([len(torch.where(tst_dataset_labels==lbl)[0]) for lbl in test_unique_lbls]).to(device).float()
        
        for i, lbl in enumerate(trn_unique_lbls):
            train_class_split_size = trn_frequencies[i] // current_num_client
            val_class_split_size = val_frequencies[i] // current_num_client
            test_class_split_size = test_frequencies[i] // current_num_client
            
            trn_data_list = torch.where(trn_dataset_labels==lbl)[0].tolist()
            val_data_list = torch.where(val_dataset_labels==lbl)[0].tolist()
            tst_data_list = torch.where(tst_dataset_labels==lbl)[0].tolist()
            
            for j in range(current_num_client):
                if j == current_num_client-1:
                    for d in trn_data_list[j*int(train_class_split_size.item()):]:
                        client_data[j]['trn']['x'].append(current_task_trn_dataset['x'][d])
                        client_data[j]['trn']['y'].append(current_task_trn_dataset['y'][d])
                    for d in val_data_list[j*int(val_class_split_size.item()):]:
                        client_data[j]['val']['x'].append(current_task_val_dataset['x'][d])
                        client_data[j]['val']['y'].append(current_task_val_dataset['y'][d])
                    for d in tst_data_list[j*int(test_class_split_size.item()):]:
                        client_data[j]['tst']['x'].append(current_task_tst_dataset['x'][d])
                        client_data[j]['tst']['y'].append(current_task_tst_dataset['y'][d])
                else:
                    for d in trn_data_list[j*int(train_class_split_size.item()):(j+1)*int(train_class_split_size.item())]:
                        client_data[j]['trn']['x'].append(current_task_trn_dataset['x'][d])
                        client_data[j]['trn']['y'].append(current_task_trn_dataset['y'][d])
                    for d in val_data_list[j*int(val_class_split_size.item()):(j+1)*int(val_class_split_size.item())]:
                        client_data[j]['val']['x'].append(current_task_val_dataset['x'][d])
                        client_data[j]['val']['y'].append(current_task_val_dataset['y'][d])
                    for d in tst_data_list[j*int(test_class_split_size.item()):(j+1)*int(test_class_split_size.item())]:
                        client_data[j]['tst']['x'].append(current_task_tst_dataset['x'][d])
                        client_data[j]['tst']['y'].append(current_task_tst_dataset['y'][d])
        
        if args.datasets == ['ham10000'] or args.datasets == ['cch5000']:
            Dataset = basedat.BaseDataset
        else:
            Dataset = memd.MemoryDataset
            
        client_loaders = []
        for i in range(current_num_client):
            train_dataset = Dataset(client_data[i]['trn'], trn_transform, class_indices) 
            valid_dataset = Dataset(client_data[i]['val'], tst_transform, class_indices) 
            test_dataset = Dataset(client_data[i]['tst'], tst_transform, class_indices) 
            client_loaders.append(make_client_loaders(train_dataset, valid_dataset, test_dataset, args.batch_size))
        
        client_models = []
        for i in range(current_num_client):
            client_models.append(global_client_net.to(device))

        if t >= max_task:
            continue

        print('*' * 108)
        print('Task {:2d}'.format(t))
        print('*' * 108)

        # Add head for current task
        if args.approach != 'der_split_dp':
            global_server_net.add_head(taskcla[t][1])
        global_server_net.to(device)
        

        # Train
        appr.train(t, client_loaders, taskcla)
        print('-' * 108)

        # Test
        for u in range(t + 1):
            if u == t:
                test_loss, acc_taw[t, u], acc_tag[t, u] = appr.eval(u, client_loaders, appr.client_model, num_clients[t], test=True)
            else:
                test_loss, acc_taw[t, u], acc_tag[t, u] = appr.eval(u, previous_client_loaders[u], appr.client_model, num_clients[t], test=True)
            
            if u < t:
                forg_taw[t, u] = acc_taw[:t, u].max(0) - acc_taw[t, u]
                forg_tag[t, u] = acc_tag[:t, u].max(0) - acc_tag[t, u]
            print('>>> Test on task {:2d} : loss={:.3f} | TAw acc={:5.1f}%, forg={:5.1f}%'
                  '| TAg acc=:{:5.1f}%, forg={:5.1f}% <<<'.format(u, test_loss,
                                                                 100 * acc_taw[t, u], 100 * forg_taw[t, u],
                                                                 100 * acc_tag[t, u], 100 * forg_tag[t, u]))
            logger.log_scalar(task=t, iter=u, name='loss', group='test', value=test_loss)
            logger.log_scalar(task=t, iter=u, name='acc_taw', group='test', value=100 * acc_taw[t, u])
            logger.log_scalar(task=t, iter=u, name='acc_tag', group='test', value=100 * acc_tag[t, u])
            logger.log_scalar(task=t, iter=u, name='forg_taw', group='test', value=100 * forg_taw[t, u])
            logger.log_scalar(task=t, iter=u, name='forg_tag', group='test', value=100 * forg_tag[t, u])
        
        # Save client loaders
        previous_client_loaders.append(client_loaders)
        
        # Save
        print('Save at ' + os.path.join(args.results_path, full_exp_name))
        logger.log_result(acc_taw, name="acc_taw", step=t)
        logger.log_result(acc_tag, name="acc_tag", step=t)
        logger.log_result(forg_taw, name="forg_taw", step=t)
        logger.log_result(forg_tag, name="forg_tag", step=t)
#         logger.save_model(net.state_dict(), task=t)
        logger.log_result(acc_taw.sum(1) / np.tril(np.ones(acc_taw.shape[0])).sum(1), name="avg_accs_taw", step=t)
        logger.log_result(acc_tag.sum(1) / np.tril(np.ones(acc_tag.shape[0])).sum(1), name="avg_accs_tag", step=t)
        aux = np.tril(np.repeat([[tdata[1] for tdata in taskcla[:max_task]]], max_task, axis=0))
        logger.log_result((acc_taw * aux).sum(1) / aux.sum(1), name="wavg_accs_taw", step=t)
        logger.log_result((acc_tag * aux).sum(1) / aux.sum(1), name="wavg_accs_tag", step=t)

    # Print Summary
    utils.print_summary(acc_taw, acc_tag, forg_taw, forg_tag)
    print('[Elapsed time = {:.1f} h]'.format((time.time() - tstart) / (60 * 60)))
    print('Done!')

    return acc_taw, acc_tag, forg_taw, forg_tag, logger.exp_path
    ####################################################################################################################


if __name__ == '__main__':
    main()
