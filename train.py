#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import fire
import sys
import random

import pydev

sys.path.append('/home/psdz/lab/practice/learn_pytorch')
import easy_train

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from torchvision.transforms import *

from models import *

def V1_transform():
    train_transform = Compose([
        RandomCrop(32, padding=4, padding_mode='reflect'), 
        RandomHorizontalFlip(),
        ToTensor(),
        RandomErasing(p=0.5, scale=(0.1, 0.1)),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    test_transform = Compose([
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    return train_transform, test_transform

def train(epoch=2, batch_size=64, data_path='../dataset/voc'):
    train_transform, test_transform = V1_transform()

    train = tv.datasets.VOCSegmentation(data_path, image_set='train')
    test = tv.datasets.VOCSegmentation(data_path, image_set='val')

    train_dataloader = torch.utils.data.DataLoader(train, shuffle=True, batch_size=batch_size, pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=32)

    # train phase.
    model = tv.models.segmentation.fcn_resnet101(pretrained=True)

    cuda = torch.device('cuda')     # Default CUDA device
    model.to(cuda)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    loss_fn = nn.CrossEntropyLoss()

    easy_train.epoch_train(train_dataloader, model, optimizer, loss_fn, epoch, 
            batch_size=batch_size, device=cuda, validation=test_dataloader, validation_epoch=3,
            scheduler=None)

    easy_train.epoch_test(test_dataloader, model, device=cuda)

    print 'train over'

if __name__=='__main__':
    fire.Fire()
