#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import torchvision as tv
import torch

class V1_tv_dlabv3_res101(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.__model = tv.models.segmentation.deeplabv3_resnet101(pretrained=True).cuda()
    
    def forward(self, x):
        y = self.__model(x)
        y = y['out'].permute(0,2,3,1)
        return y

class V0_tv_fcn_res101(torch.nn.Module):
    '''
        MIoU 67.97%
    '''
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.__model = tv.models.segmentation.fcn_resnet101(pretrained=True).cuda()
    
    def forward(self, x):
        y = self.__model(x)
        y = y['out'].permute(0,2,3,1)
        return y


