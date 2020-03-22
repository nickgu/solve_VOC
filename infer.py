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

import train
import models

def infer(data_path, idx=0, image_set='val', model_path=None):


    ori_imgs = tv.datasets.VOCSegmentation(data_path, image_set=image_set)
    img_tr, tgt_tr = train.V0_transform()
    data = tv.datasets.VOCSegmentation(data_path, image_set=image_set, transform=img_tr, target_transform=tgt_tr)
    pydev.info('Data loaded')

    model = models.V0_torchvision_fcn_res101()
    pydev.info('Model loaded')
    if model_path:
        pydev.info('load model params from [%s]' % model_path)
        model.load_state_dict(torch.load(model_path))
        pydev.info('model load ok')

    y = model(data[idx][0].unsqueeze(0).cuda())

    out = y.squeeze()
    v = out.max(dim=2).indices.cpu().to(torch.uint8)

    c_union = sum( (v>0) + (data[idx][1]) )
    print sum(c_union)
    c_inter = sum( (v>0) * (v==data[idx][1]) )
    print sum(c_inter)
    precision = sum(c_inter)*1. / sum(c_union)
    pydev.info('Precision=%.2f%%' % (precision*100.))

    im = PIL.Image.fromarray(v.numpy())
    im.putpalette(ori_imgs[0][1].getpalette())

    ori_imgs[idx][0].show()
    ori_imgs[idx][1].show()
    im.show()


if __name__=='__main__':
    fire.Fire(infer)
