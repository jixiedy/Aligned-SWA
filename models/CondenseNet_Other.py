#coding=utf-8
from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.nn import functional as F
import json
from easydict import EasyDict as edict
import numpy as np

from config.CondenseNet_Other.weights_initializer import init_model_weights
from config.CondenseNet_Other.denseblock import DenseBlock
from config.CondenseNet_Other.layers import LearnedGroupConv
from aligned.HorizontalMaxPool2D import HorizontalMaxPool2d

"""
CondenseNet Model
name: condensenet.py
date: May 2018
"""

__all__ = ['CondenseNet_Other']

class CondenseNet_Other(nn.Module):
    # def __init__(self, args, num_classes, **kwargs):
    def __init__(self, args, num_classes, loss={'softmax'}, stages=[4,6,8], growth=[8,16,32], aligned=False, **kwargs):
        super(CondenseNet_Other, self).__init__()
        self.args = args
        self.loss = loss

        self.stages = args.stages
        self.growth = args.growth
        assert len(self.stages) == len(self.growth)

        # self.init_stride = args.init_stride
        # self.pool_size = args.pool_size
        self.num_classes = num_classes

        self.progress = 0.0
        self.num_filters = 2 * self.growth[0]
        """
        Initializing layers
        """
        self.transition_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.pool = nn.AvgPool2d(8)
        self.relu = nn.ReLU(inplace=True)

        # self.init_conv = nn.Conv2d(in_channels=self.args.input_channels, out_channels=self.num_filters, kernel_size=3, stride=self.init_stride, padding=1, bias=False)
        self.init_conv = nn.Conv2d(3, out_channels=self.num_filters, kernel_size=3, stride=1, padding=1, bias=False)

        self.denseblock_one = DenseBlock(num_layers=self.stages[0], in_channels= self.num_filters, growth=self.growth[0], args=args)

        self.num_filters += self.stages[0] * self.growth[0]

        self.denseblock_two = DenseBlock(num_layers=self.stages[1], in_channels= self.num_filters, growth=self.growth[1], args=args)

        self.num_filters += self.stages[1] * self.growth[1]

        self.denseblock_three = DenseBlock(num_layers=self.stages[2], in_channels= self.num_filters, growth=self.growth[2], args=args)

        self.num_filters += self.stages[2] * self.growth[2]
        self.batch_norm = nn.BatchNorm2d(self.num_filters)

        self.classifier = nn.Linear(self.num_filters, self.num_classes)

        self.apply(init_model_weights)

        self.feat_dim = self.num_filters  # feature dimension
        self.aligned = aligned
        self.horizon_pool = HorizontalMaxPool2d()
        if self.aligned:
            self.bn = nn.BatchNorm2d(self.num_filters)
            self.relu = nn.ReLU(inplace=True)
            self.conv1 = nn.Conv2d(self.num_filters, 128, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x, progress=None):
        if progress:
            LearnedGroupConv.global_progress = progress

        x = self.init_conv(x)

        x = self.denseblock_one(x)
        x = self.transition_pool(x)

        x = self.denseblock_two(x)
        x = self.transition_pool(x)

        x = self.denseblock_three(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.pool(x)

        if not self.training:
            lf = self.horizon_pool(x)
        if self.aligned:
            lf = self.bn(x)
            lf = self.relu(lf)
            lf = self.horizon_pool(lf)
            lf = self.conv1(lf)
        if self.aligned or not self.training:
            lf = lf.view(lf.size()[0:3])
            lf = lf / torch.pow(lf, 2).sum(dim=1, keepdim=True).clamp(min=1e-12).sqrt()
        x = F.avg_pool2d(x, x.size()[2:])
        # 将前面多维度的tensor展平成一维,x.size(0)指batchsize的值,batchsize指转换后有几行，
        # 而-1指在不告诉函数有多少列的情况下，根据原tensor数据和batchsize自动分配列数
        f = x.view(x.size(0), -1)
        # f = 1. * f / (torch.norm(f, 2, dim=-1, keepdim=True).expand_as(f) + 1e-12)
        if not self.training:
            return f, lf
        y = self.classifier(f)
        if self.loss == {'softmax'}:
            return y
        elif self.loss == {'metric'}:
            if self.aligned: return f, lf
            return f
        elif self.loss == {'softmax', 'metric'}:
            if self.aligned: return y, f, lf
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

        # x = x.view(x.size(0), -1)

        # out = self.classifier(x)

        # return out

"""
#########################
Model Architecture:
#########################

Input: (N, 32, 32, 3)

- Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
DenseBlock(num_layers=14, in_channels=16, growth=8)
- AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False, count_include_pad=True)
DenseBlock(num_layers=14, in_channels=128, growth=16)
- AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False, count_include_pad=True)
DenseBlock(num_layers=14, in_channels=352, growth=32)
- BatchNorm2d(800, eps=1e-05, momentum=0.1, affine=True)
- ReLU(inplace)
- AvgPool2d(kernel_size=8, stride=8, padding=0, ceil_mode=False, count_include_pad=True)
- Linear(in_features=800, out_features=10, bias=True)
"""
