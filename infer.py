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
import tqdm

import train
import models

def precision(y_, y):
    c_union = sum( ((y>0)*(y<=30)) + (y_>0)*(y<=30) )
    c_inter = sum( (y>0) * (y==y_) )
    precision = sum(c_inter)*1. / sum(c_union)
    return precision

def infer(data_path, image_set='val', model_path=None):
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

    while True:
        idx = sys.stdin.readline()
        idx = int(idx)
        pydev.info('Index=%d' % idx)

        y = model(data[idx][0].unsqueeze(0).cuda())

        out = y.squeeze()
        v = out.max(dim=2).indices.cpu().to(torch.uint8)
        prec = precision(v, data[idx][1])
        pydev.info('Precision=%.2f%%' % (prec*100.))

        im = PIL.Image.fromarray(v.numpy())
        im.putpalette(ori_imgs[0][1].getpalette())

        ori_imgs[idx][0].show()
        ori_imgs[idx][1].show()
        im.show()

def eval(data_path, image_set='val', model_path=None):
    img_tr, tgt_tr = train.eval_transform()
    data = tv.datasets.VOCSegmentation(data_path, image_set=image_set, transform=img_tr, target_transform=tgt_tr)
    pydev.info('Data loaded')

    model = models.V0_torchvision_fcn_res101()
    pydev.info('Model loaded')
    if model_path:
        pydev.info('load model params from [%s]' % model_path)
        model.load_state_dict(torch.load(model_path))
        pydev.info('model load ok')

    with torch.no_grad():
        model.eval()
        bar = tqdm.tqdm(range(len(data)))
        acc_prec = 0
        acc_count = 0
        for idx in bar:
            y = model(data[idx][0].unsqueeze(0).cuda())

            out = y.squeeze()
            v = out.max(dim=2).indices.cpu().to(torch.uint8)
            prec = precision(v, data[idx][1])
            acc_prec += prec
            acc_count += 1

            bar.set_description('MeanPrecision=%.2f%%' % (acc_prec*100. / acc_count))
        pydev.info('MeanPrecision=%.2f%%' % (acc_prec*100. / acc_count))

if __name__=='__main__':
    fire.Fire()
