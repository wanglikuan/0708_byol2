import torch
from flcore.clients.byol_pytorch import BYOL ####
from torchvision import models
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision

import json
import numpy as np
import os
import torch.nn as nn
import torchvision.transforms as transforms
from tqdm import trange
import random


def get_data_dir(dataset): ### dataset:str ### for test use [Cifar10-class1]
    if 'EMnist' in dataset:
        dataset_ = dataset.replace('class', '').split('-')
        # path_prefix=os.path.join('data', 'Mnist', 'u20alpha{}min10ratio{}'.format(alpha, ratio))
        if 'mtl' in dataset:
            classes = dataset_[2]
            path_prefix = os.path.join('data', 'EMnist', 'u20c10-mtl-class{}'.format(classes))
        else:
            classes = dataset_[1]
            path_prefix = os.path.join('data', 'EMnist', 'u20c10-class{}'.format(classes))
        train_data_dir = os.path.join(path_prefix, 'train')
        test_data_dir = os.path.join(path_prefix, 'test')
        proxy_data_dir = 'data/proxy_data/mnist-n10/'

    elif 'Cifar10' in dataset:
        dataset_ = dataset.replace('class', '').split('-')
        # path_prefix=os.path.join('data', 'Mnist', 'u20alpha{}min10ratio{}'.format(alpha, ratio))
        if 'mtl' in dataset:
            classes = dataset_[2]
            path_prefix = os.path.join('data', 'CIFAR10', 'u20c10-mtl-class{}'.format(classes))
        else:
            classes = dataset_[1]
            path_prefix = os.path.join('data', 'CIFAR10', 'u10c10-class{}'.format(classes)) ###---###---###---### KEY LINE ###---###---###---###
        train_data_dir = os.path.join(path_prefix, 'train')
        test_data_dir = os.path.join(path_prefix, 'test')
        proxy_data_dir = 'data/proxy_data/mnist-n10/'

    elif 'celeb' in dataset.lower():
        dataset_ = dataset.lower().replace('user', '').replace('agg','').split('-')
        user, agg_user = dataset_[1], dataset_[2]
        path_prefix = os.path.join('data', 'CelebA', 'user{}-agg{}'.format(user,agg_user))
        train_data_dir=os.path.join(path_prefix, 'train')
        test_data_dir=os.path.join(path_prefix, 'test')
        proxy_data_dir=os.path.join('/user500/', 'proxy')

    else:
        raise ValueError("Dataset not recognized.")
    return train_data_dir, test_data_dir, proxy_data_dir


def read_data(dataset): ### dataset:str ###
    train_data_dir, test_data_dir, proxy_data_dir = get_data_dir(dataset)
    clients = []
    groups = []
    train_data = {}
    test_data = {}
    proxy_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json') or f.endswith(".pt")]
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        if file_path.endswith("json"):
            with open(file_path, 'r') as inf:
                cdata = json.load(inf)
        elif file_path.endswith(".pt"):
            with open(file_path, 'rb') as inf:
                cdata = torch.load(inf)
        else:
            raise TypeError("Data format not recognized: {}".format(file_path))

        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    clients = list(sorted(train_data.keys()))

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json') or f.endswith(".pt")]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        if file_path.endswith(".pt"):
            with open(file_path, 'rb') as inf:
                cdata = torch.load(inf)
        elif file_path.endswith(".json"):
            with open(file_path, 'r') as inf:
                cdata = json.load(inf)
        else:
            raise TypeError("Data format not recognized: {}".format(file_path))
        test_data.update(cdata['user_data'])


    if proxy_data_dir and os.path.exists(proxy_data_dir):
        proxy_files=os.listdir(proxy_data_dir)
        proxy_files=[f for f in proxy_files if f.endswith('.json') or f.endswith(".pt")]
        for f in proxy_files:
            file_path=os.path.join(proxy_data_dir, f)
            if file_path.endswith(".pt"):
                with open(file_path, 'rb') as inf:
                    cdata=torch.load(inf)
            elif file_path.endswith(".json"):
                with open(file_path, 'r') as inf:
                    cdata=json.load(inf)
            else:
                raise TypeError("Data format not recognized: {}".format(file_path))
            proxy_data.update(cdata['user_data'])

    return clients, groups, train_data, test_data, proxy_data

def convert_data_contrastive(X1, X2, y, dataset=''):
    if not isinstance(X1, torch.Tensor):
        if 'celeb' in dataset.lower():
            X1=torch.Tensor(X1).type(torch.float32).permute(0, 3, 1, 2)
            X2=torch.Tensor(X2).type(torch.float32).permute(0, 3, 1, 2)
            y=torch.Tensor(y).type(torch.int64)

        else:
            X1=torch.Tensor(X1).type(torch.float32)
            X2=torch.Tensor(X2).type(torch.float32)
            y=torch.Tensor(y).type(torch.int64)
    return X1, X2, y

def read_user_data_contrastive(index, data, dataset='', count_labels=False):
    #data contains: clients, groups, train_data, test_data, proxy_data(optional)
    id = data[0][index]
    train_data = data[2][id]
    test_data = data[3][id]
    X_train1, X_train2, y_train = convert_data_contrastive(train_data['x1'], train_data['x2'], train_data['y'], dataset=dataset)
    train_data = [((x1, x2), y) for x1, x2, y in zip(X_train1, X_train2, y_train)]
    X_test1, X_test2, y_test = convert_data_contrastive(test_data['x1'], test_data['x2'], test_data['y'], dataset=dataset)
    test_data = [((x1, x2), y) for x1, x2, y in zip(X_test1, X_test2, y_test)]
    if count_labels:
        label_info = {}
        unique_y, counts=torch.unique(y_train, return_counts=True)
        unique_y=unique_y.detach().numpy()
        counts=counts.detach().numpy()
        label_info['labels']=unique_y
        label_info['counts']=counts
        return id, train_data, test_data, label_info
    return id, train_data, test_data

#################################### 上面都是 model_util.py 的代码 ################################################


def test_and_save(label_id, client_id, data, test_sample_num=100): ### label_id是要跑的data的label；client_id是受测试的client的id ###
    device = "cuda"
    id, train_data , test_data = read_user_data_contrastive(label_id, data, dataset='Cifar10-class1') #输出的 train 为指定label数据
    trainloader = DataLoader(train_data, test_sample_num, drop_last=True)
    iter_trainloader = iter(trainloader)
########## get_next_train_batch() ########## 准备数据
    try:
        # Samples a new batch for persionalizing
        ((x_i, x_j), y) = next(iter_trainloader)
    except StopIteration:
        # restart the generator if the previous generator is exhausted.
        iter_trainloader = iter(trainloader)
        ((x_i, x_j), y) = next(iter_trainloader)
    if type(x_i) == type([]):
        x_i[0] = x_i[0].to(device)
        x_j[0] = x_j[0].to(device)        
    else:
        x_i = x_i.to(device)
        x_j = x_j.to(device)        
    y = y.to(device)
########## return x_i, x_j, y ##########

########## 准备网络 ##########
    resnet_18 = models.resnet18(pretrained=True)
    resnet_18.load_state_dict(torch.load('net_client{}.pt'.format(client_id), map_location='cpu')) ### 加载指定模型参数 ###
    Model = resnet_18.to(device)
########## return Model #####

    learner = BYOL(
        Model,
        image_size = 32,
        hidden_layer = 'avgpool'
    )

    projection, embedding = learner(x_i, x_j, return_embedding = True)

    result_data = {'user': client_id, 'label': label_id, 'embedding': {}}
    result_data['embedding'] = {
        'projection': projection,
        'representation': embedding}

    print('projection.shape: ', projection.shape, '    representation.shape: ', embedding.shape)

########## 保存结果 ##########
    data_path = f'./get-representation-result'
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    data_path=os.path.join(data_path, "label{}_client{}_num{}.pt".format(label_id,client_id,test_sample_num))
    with open(data_path, 'wb') as outfile:
        print(f"Dumping train data => {data_path}")
        torch.save(result_data, outfile)
##########

if __name__ == "__main__":
    data = read_data('Cifar10-class1')
    test_labels = [1,2]
    test_clients = [2,3]
    test_sample_num = 100 ### num for single label ###
    for label in test_labels:
        for client in test_clients:
            test_and_save(label, client, data, test_sample_num)
