#coding=utf-8
from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from aligned.HorizontalMaxPool2D import HorizontalMaxPool2d
# from util.FeatureExtractor import FeatureExtractor

__all__ = ['DenseNet121', 'DenseNet161', 'DenseNet169', 'DenseNet201']

class DenseNet121(nn.Module):
    def __init__(self, num_classes, loss={'softmax'}, aligned=False, **kwargs):
        super(DenseNet121, self).__init__() 
        self.loss = loss
        densenet121 = torchvision.models.densenet121(pretrained=True)
        self.base = densenet121.features
        self.classifier = nn.Linear(1024, num_classes)
        self.feat_dim = 1024 # feature dimension
        self.aligned = aligned
        self.horizon_pool = HorizontalMaxPool2d()
        if self.aligned:
            self.bn = nn.BatchNorm2d(1024)
            self.relu = nn.ReLU(inplace=True)
            self.conv1 = nn.Conv2d(1024, 128, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.base(x) # self.features
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

class DenseNet161(nn.Module):
    def __init__(self, num_classes, loss={'softmax'}, aligned=False, **kwargs):
        super(DenseNet161, self).__init__() 
        self.loss = loss
        densenet161 = torchvision.models.densenet161(pretrained=True)
        self.base = densenet161.features
        self.classifier = nn.Linear(2208, num_classes)
        self.feat_dim = 2208 # feature dimension
        self.aligned = aligned
        self.horizon_pool = HorizontalMaxPool2d()
        if self.aligned:
            self.bn = nn.BatchNorm2d(2208)
            self.relu = nn.ReLU(inplace=True)
            self.conv1 = nn.Conv2d(2208, 128, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.base(x)
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

class DenseNet169(nn.Module):
    def __init__(self, num_classes, loss={'softmax'}, aligned=False, **kwargs):
        super(DenseNet169, self).__init__() 
        self.loss = loss
        densenet169 = torchvision.models.densenet169(pretrained=True)
        self.base = densenet169.features
        self.classifier = nn.Linear(1664, num_classes)
        self.feat_dim = 1664 # feature dimension
        self.aligned = aligned
        self.horizon_pool = HorizontalMaxPool2d()
        if self.aligned:
            self.bn = nn.BatchNorm2d(1664)
            self.relu = nn.ReLU(inplace=True)
            self.conv1 = nn.Conv2d(1664, 128, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.base(x)
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

class DenseNet201(nn.Module):
    def __init__(self, num_classes, loss={'softmax'}, aligned=False, **kwargs):
        super(DenseNet201, self).__init__() 
        self.loss = loss
        densenet201 = torchvision.models.densenet201(pretrained=True)
        self.base = densenet201.features
        self.classifier = nn.Linear(1920, num_classes)
        self.feat_dim = 1920 # feature dimension
        self.aligned = aligned
        self.horizon_pool = HorizontalMaxPool2d()
        if self.aligned:
            self.bn = nn.BatchNorm2d(1920)
            self.relu = nn.ReLU(inplace=True)
            self.conv1 = nn.Conv2d(1920, 128, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.base(x)
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
