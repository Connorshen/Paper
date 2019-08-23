# -*- coding: utf-8 -*-
"""
@Time    : 2019/8/22 下午7:42
@Author  : 比尔丶盖子
@Email   : 914138410@qq.com
"""
import torch.nn.functional as F
from torch import nn
import torch


class Output(nn.Module):
    def __init__(self, in_features, out_features, synaptic_th=0.8):
        """
        :param in_features: 输入通道数字
        :param out_features: 输出通道数字
        """
        super(Output, self).__init__()
        self.synaptic_th = synaptic_th
        self.weight = torch.rand(out_features, in_features).cuda()

    def forward(self, x):
        x = F.linear(x, self.filter(self.weight, self.synaptic_th))
        x = x - torch.mean(x, dim=1, keepdim=True).repeat(1, x.shape[1])
        return x

    @staticmethod
    def filter(weight, synaptic_th):
        weight_filter = torch.zeros(weight.shape).cuda()
        weight_filter[weight > synaptic_th] = 1
        return weight_filter
