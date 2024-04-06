# -*- coding: utf-8 -*-
import torch
from networks.resnet32_split import *
from networks.resnet18_split import *
import datasets.memory_dataset as memd
import datasets.base_dataset as basedat
from networks.network import LLL_Net
from datasets.data_loader_split import get_final_datas, make_client_loaders
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle
from datasets.exemplars_selection_split import override_dataset_transform
import numpy as np
import math

# CUDA device 설정
torch.cuda.set_device(0)
device = 'cuda'


def compute_mean_of_exemplars(server_model, client_model, dataset, transform, current_cls, plot_nums=None):
    cls_protos = []
    cls_feats_list = []
    # change transforms to evaluation for this calculation
    with override_dataset_transform(dataset, transform) as _ds:
        # change dataloader so it can be fixed to go sequentially (shuffle=False), this allows to keep same order
        icarl_loader = DataLoader(_ds, batch_size=128, shuffle=False, num_workers=2)
        # extract features from the model for all train samples
        # Page 2: "All feature vectors are L2-normalized, and the results of any operation on feature vectors,
        # e.g. averages are also re-normalized, which we do not write explicitly to avoid a cluttered notation."
        extracted_features = []
        extracted_targets = []
        with torch.no_grad():
            server_model.eval()
            client_model.eval()
            for images, targets in icarl_loader:
                client_feats = client_model(images.to(device))
                feats = server_model(client_feats, return_features=True)[1]
                # normalize
                extracted_features.append(feats / feats.norm(dim=1).view(-1, 1))
                extracted_targets.extend(targets)
        extracted_features = torch.cat(extracted_features)
        extracted_targets = np.array(extracted_targets)
        for curr_cls in current_cls:
            # get all indices from current class
            cls_ind = np.where(extracted_targets == curr_cls)[0]
            # get all extracted features for current class
            cls_feats = extracted_features[cls_ind]
            # add the exemplars to the set and normalize
            cls_feats_mean = cls_feats.mean(0) / cls_feats.mean(0).norm()
            cls_protos.append(cls_feats_mean.tolist())
    return cls_protos


# +
def main(dataset_name, pl_nums):
    dataset = dataset_name
    print(f"Current dataset: {dataset}")
    
    if dataset in ['bloodmnist', 'tissuemnist', 'cch5000']:
        # bloodmnist, tissuemnist, cch5000 task _classes
        task_classes={'0': [0,1], '1':[0,1,2,3], '2':[0,1,2,3,4,5], '3':[0,1,2,3,4,5,6,7]}
        n_tasks = 4
    elif dataset in ['pathmnist']:
        # pathmnist task _classes
        task_classes={'0': [0,1,2], '1':[0,1,2,3,4], '2':[0,1,2,3,4,5,6], '3':[0,1,2,3,4,5,6,7,8]}
        n_tasks = 4
    elif dataset in ['organamnist']:
        # organmnist task _classes
        task_classes={'0': [0,1,2], '1':[0,1,2,3,4,5], '2':[0,1,2,3,4,5,6,7,8], '3':[0,1,2,3,4,5,6,7,8,9,10]}
        n_tasks = 4
    elif dataset in ['svhn']:
        # svhn task _classes
        task_classes={'0': [0,1], '1':[0,1,2,3], '2':[0,1,2,3,4,5], '3':[0,1,2,3,4,5,6,7], '4':[0,1,2,3,4,5,6,7,8,9]}
        n_tasks = 5
    elif dataset in ['cifar100_icarl']:
        # cifar100 task _classes
        task_classes={'0': list(range(10)), '1':list(range(20)), '2':list(range(30)), '3':list(range(40)), \
                      '4':list(range(50)), '5': list(range(60)), '6': list(range(70)), '7': list(range(80)), \
                     '8': list(range(90)), '9': list(range(100))}
        n_tasks = 10
    elif dataset in ['ham10000']:
        # ham10000 task _classes
        task_classes={'0': [0,1,2], '1':[0,1,2,3,4], '2':[0,1,2,3,4,5,6]}
        n_tasks = 3
    else:
        print(f"{dataset} is not used dataset!!")
        exit(True)
        
    # test dataset 불러오기
    # tst_data:task 별로 구성됨
    # 하나의 task에는 image ['x'], label['y'] 딕셔너리로 구성됨
    _, _, tst_data, taskcla, class_indices, _, tst_transform \
    = get_final_datas([dataset], n_tasks, None, \
                  128, num_workers=2, \
                  pin_memory=False)
    
    # Task 별로 클래스별 클래스 프로토타입까지의 평균 거리 계산
    plot_nums = pl_nums
    for s in [0,1,2,3,4]:
        print(f"Seed {s}")
        
        # 모델 선언
        if dataset in ['cch5000', 'ham10000']:
            init_server_model = ResNet18_server_side(Basicblock_resnet18, [2,2,2]).to(device)
            server_model = LLL_Net(init_server_model, remove_existing_head=not False)
            client_model = ResNet18_client_side(Basicblock_resnet18, input_channel=3).to(device)
            Dataset = basedat.BaseDataset

        else:
            init_server_model = ResNet32_server_side(BasicBlock, [5,5]).to(device)
            server_model = LLL_Net(init_server_model, remove_existing_head=not False)
            if dataset in ['organamnist', 'tissuemnist']:
                client_model = ResNet32_client_side(BasicBlock, [5], 1).to(device)
            else:
                client_model = ResNet32_client_side(BasicBlock, [5], 3).to(device)
            Dataset = memd.MemoryDataset
            
        txt_result = open(f"{dataset} task seed{s} class dist result.txt","w+")
        print(f"plot nums: {plot_nums}")
        txt_result.write(f"plot nums: {plot_nums} \n")
        for t in range(n_tasks):
            print(f"----------------------------Task {t}----------------------------")
            txt_result.write(f"----------------------------Task {t}---------------------------- \n")
            tst_datas = []
            before_dists = []
            after_dists = []
            for tt in range(t+1):
                tst_datas.append(Dataset(tst_data[tt], tst_transform, class_indices))

            server_model.add_head(taskcla[t][1])
            server_model.to(device)
            #-----------------------------------before Align-----------------------------------#
            print("Before Align")
            txt_result.write("Before Align \n")

            # model load
            server_model.load_state_dict(torch.load(f'./model_save/{dataset}/seed{s}/task{t}_before_align_server_model.pt'))
            client_model.load_state_dict(torch.load(f'./model_save/{dataset}/seed{s}/task{t}_before_align_client_model.pt'))
            server_model.eval()
            client_model.eval()

            # current task exemplar load
            task_exemplars = torch.load(f'./model_save/{dataset}/seed{s}/task{t}_before_align_exemplars.pt')

            # compute exemplar class prototypes
            # prototype은 학습한 모든 class 출력
            cls_protos = compute_mean_of_exemplars(server_model, client_model, task_exemplars, tst_transform, task_classes[f'{t}'])
            cls_protos = np.array(cls_protos)
            print("cls_protos: ", cls_protos.shape)

            # test data의 feature 추출
            extracted_features = []
            extracted_targets = []
            cls_feats_list = []
            for tt in range(t+1):
                test_dataloader = DataLoader(tst_datas[tt], batch_size=128, shuffle=False, num_workers=2)
                with torch.no_grad():
                    server_model.eval()
                    client_model.eval()
                    for images, targets in test_dataloader:
                        client_feats = client_model(images.to(device))
                        feats = server_model(client_feats, return_features=True)[1]
                        # normalize
                        extracted_features.append(feats / feats.norm(dim=1).view(-1, 1))
                        extracted_targets.extend(targets)
            extracted_features = torch.cat(extracted_features)
            extracted_targets = np.array(extracted_targets)
            n_datas_per_cls = [] # 클래스별 test data 개수 저장하는 리스트
            for curr_cls in task_classes[f'{t}']:
                # get all indices from current class
                cls_ind = np.where(extracted_targets == curr_cls)[0]
                print(curr_cls, " ind len: ", cls_ind.shape)
                # get all extracted features for current class
                cls_feats = extracted_features[cls_ind]
                if plot_nums == -1:
                    n_datas_per_cls.append(cls_ind.shape[0])
                    cls_feats_list.extend(deepcopy(cls_feats.cpu().numpy()))
                else:
                    cls_feats_list.extend(deepcopy(cls_feats[0:plot_nums].cpu().numpy()))

            cls_feats_final = np.array(cls_feats_list)
            print("cls_feats_final: ", cls_feats_final.shape)

            cls_protos = cls_protos.tolist()
            for i,proto in enumerate(cls_protos):
                cls_protos[i] = torch.tensor(proto)

            # 클래스별 각자의 프로토타입 까지의 평균 거리 계산후 txt 저장
            prev_cls_num = 0
            before_novel_avg_dist = 0
            before_novel_avg_acc = 0
            before_accs = []
            for pr in task_classes[f'{t}']:
                if plot_nums == -1:
                    if pr == 0:
                        curr_cls_feats = cls_feats_final[0:n_datas_per_cls[pr]]
                        prev_cls_num += n_datas_per_cls[pr]
                    else:
                        curr_cls_feats = cls_feats_final[prev_cls_num:prev_cls_num+n_datas_per_cls[pr]]
                        prev_cls_num += n_datas_per_cls[pr]
                else:
                    curr_cls_feats = cls_feats_final[pr*plot_nums:(pr+1)*plot_nums]
                print(f"{pr} class feats shape: ", curr_cls_feats.shape)
                means = torch.stack(cls_protos)
                means = torch.stack([means] * curr_cls_feats.shape[0])
                means = means.transpose(1, 2)
                features = torch.from_numpy(curr_cls_feats).unsqueeze(2)
                features = features.expand_as(means)
                dists = (features - means).pow(2).sum(1).squeeze()
                
                # 현재 클래스에 대한 test data 정확도
                pred = dists.argmin(1)
                hits_tag = (pred == pr).float().sum().item()
                test_acc = (hits_tag / n_datas_per_cls[pr]) * 100
                before_accs.append(test_acc)
                if (t > 0 and pr not in task_classes[f'{t-1}']) or (t == 0):
                    before_novel_avg_acc += test_acc
                    
                dist_mean = dists[:,pr].mean().item()
                print(f"{pr} class avg dist: ", dist_mean)
                txt_result.write(f"{pr} class avg dist: {dist_mean} \n")
                before_dists.append(dist_mean)
                if (t > 0 and pr not in task_classes[f'{t-1}']) or (t == 0):
                    before_novel_avg_dist += dist_mean
            
            if t == 0:
                before_novel_avg_dist = before_novel_avg_dist / len(task_classes[f'{t}'])
                before_novel_avg_acc = round(before_novel_avg_acc / len(task_classes[f'{t}']), 3)
            else:
                before_novel_avg_dist = before_novel_avg_dist / (len(task_classes[f'{t}']) - len(task_classes[f'{t-1}']))
                before_novel_avg_acc = round(before_novel_avg_acc / (len(task_classes[f'{t}']) - len(task_classes[f'{t-1}'])), 3)
                
            before_avg_dist = round(sum(before_dists)/len(before_dists), 3)
            print(f"총 평균 거리: {before_avg_dist}")
            #-----------------------------------after Align-----------------------------------#
            print()
            print("After Align")
            txt_result.write("After Align \n")
            # model load

            server_model.load_state_dict(torch.load(f'./model_save/{dataset}/seed{s}/task{t}_after_align_server_model.pt'))
            client_model.load_state_dict(torch.load(f'./model_save/{dataset}/seed{s}/task{t}_after_align_client_model.pt'))
            server_model.eval()
            client_model.eval()

            # current task exemplar load

            task_exemplars = torch.load(f'./model_save/{dataset}/seed{s}/task{t}_after_align_exemplars.pt')

            # compute exemplar class prototypes
            # prototype은 학습한 모든 class 출력
            cls_protos = compute_mean_of_exemplars(server_model, client_model, task_exemplars, tst_transform, task_classes[f'{t}'])
            cls_protos = np.array(cls_protos)
            print("cls_protos: ", cls_protos.shape)

            # test data의 feature 추출
            extracted_features = []
            extracted_targets = []
            cls_feats_list = []
            for tt in range(t+1):
                test_dataloader = DataLoader(tst_datas[tt], batch_size=128, shuffle=False, num_workers=2)
                with torch.no_grad():
                    server_model.eval()
                    client_model.eval()
                    for images, targets in test_dataloader:
                        client_feats = client_model(images.to(device))
                        feats = server_model(client_feats, return_features=True)[1]
                        # normalize
                        extracted_features.append(feats / feats.norm(dim=1).view(-1, 1))
                        extracted_targets.extend(targets)
            extracted_features = torch.cat(extracted_features)
            extracted_targets = np.array(extracted_targets)
            n_datas_per_cls = [] # 클래스별 test data 개수 저장하는 리스트
            for curr_cls in task_classes[f'{t}']:
                # get all indices from current class
                cls_ind = np.where(extracted_targets == curr_cls)[0]
                print(curr_cls, " ind len: ", cls_ind.shape)
                # get all extracted features for current class
                cls_feats = extracted_features[cls_ind]
                if plot_nums == -1:
                    n_datas_per_cls.append(cls_ind.shape[0])
                    cls_feats_list.extend(deepcopy(cls_feats.cpu().numpy()))
                else:
                    cls_feats_list.extend(deepcopy(cls_feats[0:plot_nums].cpu().numpy()))

            cls_feats_final = np.array(cls_feats_list)
            print("cls_feats_final: ", cls_feats_final.shape)

            cls_protos = cls_protos.tolist()
            for i,proto in enumerate(cls_protos):
                cls_protos[i] = torch.tensor(proto)

            # 클래스별 각자의 프로토타입 까지의 평균 거리 계산후 txt 저장
            prev_cls_num = 0
            after_novel_avg_dist = 0
            after_novel_avg_acc = 0
            after_accs = []
            for pr in task_classes[f'{t}']:
                if plot_nums == -1:
                    if pr == 0:
                        curr_cls_feats = cls_feats_final[0:n_datas_per_cls[pr]]
                        prev_cls_num += n_datas_per_cls[pr]
                    else:
                        curr_cls_feats = cls_feats_final[prev_cls_num:prev_cls_num+n_datas_per_cls[pr]]
                        prev_cls_num += n_datas_per_cls[pr]
                else:
                    curr_cls_feats = cls_feats_final[pr*plot_nums:(pr+1)*plot_nums]
#                 print(f"{pr} class feats shape: ", curr_cls_feats.shape)
                means = torch.stack(cls_protos)
                means = torch.stack([means] * curr_cls_feats.shape[0])
                means = means.transpose(1, 2)
                features = torch.from_numpy(curr_cls_feats).unsqueeze(2)
                features = features.expand_as(means)
                dists = (features - means).pow(2).sum(1).squeeze()
                
                # 현재 클래스에 대한 test data 정확도
                pred = dists.argmin(1)
                hits_tag = (pred == pr).float().sum().item()
                test_acc = (hits_tag / n_datas_per_cls[pr]) * 100
                after_accs.append(test_acc)
                if (t > 0 and pr not in task_classes[f'{t-1}']) or (t == 0):
                    after_novel_avg_acc += test_acc
                
                # 클래스 프로토타입과의 거리 평균
                dist_mean = dists[:,pr].mean().item()
                print(f"{pr} dist mean: ", dist_mean)
                txt_result.write(f"{pr} class avg dist: {dist_mean} \n")
                after_dists.append(dist_mean)
                if (t > 0 and pr not in task_classes[f'{t-1}']) or (t == 0):
                    after_novel_avg_dist += dist_mean
            
            if t == 0:
                after_novel_avg_dist = after_novel_avg_dist / len(task_classes[f'{t}'])
                after_novel_avg_acc = round(after_novel_avg_acc / len(task_classes[f'{t}']), 3)
            else:
                after_novel_avg_dist = after_novel_avg_dist / (len(task_classes[f'{t}']) - len(task_classes[f'{t-1}']))
                after_novel_avg_acc = round(after_novel_avg_acc / (len(task_classes[f'{t}']) - len(task_classes[f'{t-1}'])), 3)
            after_avg_dist = round(sum(after_dists)/len(after_dists), 3)
            print(f"총 평균 거리: {after_avg_dist}")
            
#             l = len(before_dists)
#             print(f"class별 거리 변화")
#             total_diffs = []
#             for cl in range(l):
#                 diff = after_dists[cl] - before_dists[cl]
#                 total_diffs.append(diff)
#                 txt_result.write(f"{cl} class 거리 변화: {diff} \n")

            print(f"총 평균 거리 변화: {round(after_avg_dist - before_avg_dist, 3)}")
            txt_result.write(f"All class 평균 거리 no_align, align, align-no_align \n")
            txt_result.write(f"{before_avg_dist} \n")
            txt_result.write(f"{after_avg_dist} \n")
            txt_result.write(f"{round(after_avg_dist - before_avg_dist, 3)} \n")
            
            txt_result.write(f"Novel class 평균 거리 no_align, align, align-no_align \n")
            txt_result.write(f"{round(before_novel_avg_dist, 3)} \n")
            txt_result.write(f"{round(after_novel_avg_dist, 3)} \n")
            txt_result.write(f"{round(after_novel_avg_dist - before_novel_avg_dist, 3)} \n")
            
            before_acc_avg = round(sum(before_accs)/len(before_accs),3)
            after_acc_avg = round(sum(after_accs)/len(after_accs),3)
            txt_result.write(f"Test data(All) 정확도 평균 no_align, align, align-no_align \n")
            txt_result.write(f"{before_acc_avg} \n")
            txt_result.write(f"{after_acc_avg} \n")
            txt_result.write(f"{after_acc_avg - before_acc_avg} \n")
            
            
            txt_result.write(f"Test data(Novel) 정확도 평균 no_align, align, align-no_align \n")
            txt_result.write(f"{before_novel_avg_acc} \n")
            txt_result.write(f"{after_novel_avg_acc} \n")
            txt_result.write(f"{after_novel_avg_acc - before_novel_avg_acc} \n")
            print(f"----------------------------Task {t} End---------------------------- \n")
            txt_result.write(f"----------------------------Task {t} End---------------------------- \n")
        txt_result.close()
# -

for dst in ['bloodmnist', 'organamnist', 'pathmnist', 'tissuemnist', 'svhn', 'cch5000', 'ham10000', 'cifar100_icarl']:
    main(dst, -1)

# +
# import torch
# a = torch.tensor([1,4,3,2,5,7,9,2,2,2,2,2])
# print((a == 2).float().sum())
