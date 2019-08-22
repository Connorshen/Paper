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
    def __init__(self, channel_in, channel_out):
        """
        :param channel_in: 输入通道数字
        :param channel_out: 输出通道数字
        """
        super(Output, self).__init__()
        self.weight = torch.randn(channel_out, channel_in).cuda()

    def forward(self, x):
        x = F.linear(x, self.weight)
        x = x - torch.mean(x, dim=1, keepdim=True).repeat(1, x.shape[1])
        return x
