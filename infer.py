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

label_count = 20

def label_precision(y_, y, stat_dict):
    mask = (y<=label_count)
    for c in range(0, label_count+1):
        c_union = ((y==c) + (y_==c)*mask).sum()
        c_inter = ((y==c) * (y_==c)).sum()

        stat_dict[c][0] += c_inter.item()
        stat_dict[c][1] += c_union.item()


def precision(y_, y):
    c_union = (((y>0)*(y<=label_count)) + (y_>0)*(y<=label_count)).sum()
    c_inter = ((y>0) * (y==y_)).sum()
    precision = c_inter*1. / c_union
    return precision

def infer(data_path, image_set='val', model_path=None):
    ori_imgs = tv.datasets.VOCSegmentation(data_path, image_set=image_set)
    img_tr, tgt_tr = train.V0_transform()
    data = tv.datasets.VOCSegmentation(data_path, image_set=image_set, transform=img_tr, target_transform=tgt_tr)
    pydev.info('Data loaded')

    model = models.V0_tv_fcn_res101()
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

    #model = models.V0_tv_fcn_res101()
    model = models.V1_tv_dlabv3_res101()
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
        stat_dict = {}
        for i in range(0, label_count+1):
            stat_dict[i] = [0, 0]
        for idx in bar:
            y = model(data[idx][0].unsqueeze(0).cuda())

            out = y.squeeze()
            v = out.max(dim=2).indices.cpu().to(torch.uint8)
            label_precision(v, data[idx][1], stat_dict)
            
            mean_iou = sum(map(lambda x:x[0]/(x[1]+1e-4), stat_dict.values())) / len(stat_dict)

            prec = precision(v, data[idx][1])
            acc_prec += prec
            acc_count += 1

            bar.set_description('Mean_IOU=%.2f%% MIoU_on_ins=%.2f%%' % (mean_iou*100., acc_prec*100. / acc_count))
        pydev.info('Mean_IOU=%.2f%% MIoU_on_ins=%.2f%%' % (mean_iou*100., acc_prec*100. / acc_count))

        for c in range(0, label_count+1):
            print 'label-%d: %.2f%% (%d/%d)' % (c, stat_dict[c][0]*100./stat_dict[c][1], stat_dict[c][0], stat_dict[c][1])


if __name__=='__main__':
    fire.Fire()



