#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import fire
import sys

import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from torchvision.transforms import *

import pydev

def V0_transform():
    img_tr = Compose([
        Resize((500, 500)),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # torchvision model normalize.
    ])
    tgt_tr = Compose([
        Resize((500, 500)),
        Lambda(lambda x:torch.tensor(np.array(x)))
    ])
    return img_tr, tgt_tr


def infer(data_path, idx=0, image_set='val'):

    model = tv.models.segmentation.fcn_resnet101(pretrained=True).cuda() 
    pydev.info('Model loaded')

    ori_imgs = tv.datasets.VOCSegmentation(data_path, image_set=image_set)
    img_tr, tgt_tr = V0_transform()
    data = tv.datasets.VOCSegmentation(data_path, image_set=image_set, transform=img_tr, target_transform=tgt_tr)
    pydev.info('Data loaded')

    y = model(data[idx][0].unsqueeze(0).cuda())

    out = y['out'].squeeze()
    v = out.permute(1,2,0).max(dim=2).indices.cpu().to(torch.uint8).numpy()

    im = PIL.Image.fromarray(v)
    im.putpalette(ori_imgs[0][1].getpalette())

    ori_imgs[idx][0].show()
    ori_imgs[idx][1].show()
    im.show()


if __name__=='__main__':
    fire.Fire()
