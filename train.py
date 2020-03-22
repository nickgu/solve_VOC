#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import fire
import sys
import random

import pydev
import easyai

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from torchvision.transforms import *

from models import *

def V0_transform():
    img_tr = Compose([
        Resize((500, 500)),
        ToTensor()
    ])
    tgt_tr = Compose([
        Resize((500, 500)),
        Lambda(lambda x:torch.tensor(np.array(x)).to(torch.long)),
        Lambda(lambda x:x.where(x<21, torch.zeros(x.shape).to(torch.long)))
    ])
    return img_tr, tgt_tr

def V1_transform():
    train_transform = Compose([
        Resize((500, 500)),
        RandomCrop(32, padding=4, padding_mode='reflect'), 
        RandomHorizontalFlip(),
        ToTensor(),
        RandomErasing(p=0.5, scale=(0.1, 0.1)),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    test_transform = Compose([
        Resize((500, 500)),
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    return train_transform, test_transform

def train(epoch=2, batch_size=16, data_path='../dataset/voc'):
    cuda = torch.device('cuda')     # Default CUDA device

    print data_path
    image_transform, target_transform = V0_transform()

    train = tv.datasets.VOCSegmentation(data_path, image_set='train', 
            transform=image_transform, target_transform=target_transform)
    test = tv.datasets.VOCSegmentation(data_path, image_set='val', 
            transform=image_transform, target_transform=target_transform)

    train_dataloader = torch.utils.data.DataLoader(train, shuffle=True, batch_size=batch_size, pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=32)

    # train phase.
    model = V0_torchvision_fcn_res101()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    def calc_loss(y_, y):
        y_ = y_.reshape(-1, y_.shape[-1])
        y = y.reshape(-1)
        return loss_fn(y_, y)

    easyai.epoch_train(train_dataloader, model, optimizer, loss_fn, epoch, 
            batch_size=batch_size, device=cuda, validation=test_dataloader, validation_epoch=3,
            scheduler=None)

    easyai.epoch_test(test_dataloader, model, device=cuda)

    print 'train over'

if __name__=='__main__':
    fire.Fire(train)
