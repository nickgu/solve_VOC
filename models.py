#! /bin/env python
# encoding=utf-8
# author: nickgu 
# 

import torchvision as tv
import torch

class V0_torchvision_fcn_res101(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.__model = tv.models.segmentation.fcn_resnet101(pretrained=True).cuda()
    
    def forward(self, x):
        y = self.__model(x)
        y = y['out'].permute(0,2,3,1)
        return y
