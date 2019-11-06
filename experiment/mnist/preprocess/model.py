# -*- coding: utf-8 -*-
"""
@Time    : 2019/9/5 下午1:39
@Author  : 比尔丶盖子
@Email   : 914138410@qq.com
"""
from torch.nn import Linear, BatchNorm2d, ReLU, MaxPool2d, Module, Sequential
from experiment.mnist.preprocess.gabor import Gabor2d


class Net(Module):

    def __init__(self, kernel_size_num, theta_num, sigma, lambd, kernel_size, theta):
        super(Net, self).__init__()
        self.conv1 = Sequential(
            Gabor2d(
                kernel_size_num=kernel_size_num,
                theta_num=theta_num,
                sigma=sigma,
                lambd=lambd,
                kernel_size=kernel_size,
                theta=theta
            )
        )

    def forward(self, x):
        x = self.conv1(x)
        return x
