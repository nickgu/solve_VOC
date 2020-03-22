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
import torchvision.transforms.functional as tvf
from torchvision.transforms import *

from models import *

def my_segmentation_transforms(image, segmentation):
    if random.random() > 0.5:
        angle = random.randint(-15, 15)
        image = tvf.rotate(image, angle)
        segmentation = tvf.rotate(segmentation, angle)

    start_x = random.random() * 0.2 * image.size[0]
    start_y = random.random() * 0.2 * image.size[1]
    image = tvf.crop(image, start_x, start_y, 300, 300)
    segmentation = tvf.crop(segmentation, start_x, start_y, 300, 300)

    #image.show()
    #segmentation.show()

    image = tvf.to_tensor(image)
    segmentation = torch.tensor(np.array(segmentation)).to(torch.long)
    segmentation = segmentation.where(segmentation<=30, 
                        torch.zeros(segmentation.shape).to(torch.long))
    return image, segmentation


def eval_transform():
    img_tr = Compose([
        ToTensor()
    ])
    tgt_tr = Compose([
        Lambda(lambda x:torch.tensor(np.array(x)).to(torch.long)),
        #Lambda(lambda x:x.where(x<=30, torch.zeros(x.shape).to(torch.long)))
    ])
    
    # Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # torchvision model normalize.
    return img_tr, tgt_tr

def train(epoch=10, batch_size=16, data_path='../dataset/voc'):
    cuda = torch.device('cuda')     # Default CUDA device
    print data_path

    train = tv.datasets.VOCSegmentation(data_path, image_set='train', 
            transforms=my_segmentation_transforms)
    test = tv.datasets.VOCSegmentation(data_path, image_set='val', 
            transform=my_segmentation_transforms)

    train_dataloader = torch.utils.data.DataLoader(train, shuffle=True, batch_size=batch_size, pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=32)

    # train phase.
    model = V0_torchvision_fcn_res101()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.CrossEntropyLoss()
    
    def save_model(epoch):
        torch.save(model.state_dict(), 
                'checkpoints/%s_%d.pkl' % (type(model).__name__, epoch))

    easyai.epoch_train(train_dataloader, model, optimizer, loss_fn, epoch, 
            batch_size=batch_size, device=cuda, post_process=save_model)
    print 'train over'


if __name__=='__main__':
    fire.Fire(train)
