# -*- coding: utf-8 -*-
"""
@Time    : 2019/8/22 上午10:56
@Author  : 比尔丶盖子
@Email   : 914138410@qq.com
"""
import torch.nn.functional as F
from torch.nn import Parameter
import torch
from torch import nn
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_filters(filters):
    rows = int(len(filters) / 8)
    cols = 8
    plt.figure(figsize=(10, 10))
    for r_i in range(rows):
        for c_i in range(cols):
            plt.subplot(rows, cols, r_i * cols + c_i + 1)
            f = filters[r_i * cols + c_i]
            f = f.reshape(f.shape[0], f.shape[1])
            plt.imshow(f)
    plt.show()


def gabor_filter(kernel_size_num, theta_num, sigma, lambd, kernel_size, theta):
    kernels = []
    for i in range(theta_num):
        dir_kernels = []
        for j in range(kernel_size_num):
            ksize = kernel_size[j]
            kernel = cv2.getGaborKernel(ksize=(ksize, ksize), sigma=sigma[j], theta=theta[j],
                                        lambd=lambd[j], gamma=0.3, psi=0, ktype=cv2.CV_32F)

            kernel = torch.tensor(kernel)
            kernel = kernel.view(1, 1, ksize, ksize)
            dir_kernels.append(kernel)
        kernels.append(dir_kernels)
    return kernels


class Gabor2d(nn.Module):
    def __init__(self, kernel_size_num, theta_num, sigma, lambd, kernel_size, theta):
        super(Gabor2d, self).__init__()
        self.theta_num = theta_num
        self.kernel_size_num = kernel_size_num
        self.kernels = gabor_filter(kernel_size_num, theta_num, sigma, lambd, kernel_size, theta)

    def forward(self, x):
        x_filters = []
        for i in range(self.theta_num):
            dir_kernels = self.kernels[i]
            dir_x_filters = []
            dir_x_filters_max = []
            for j in range(self.kernel_size_num):
                kernel = dir_kernels[j]
                ksize = kernel.shape[2]
                x_filter = F.conv2d(x, kernel, stride=1, padding=int((ksize - 1) / 2))
                dir_x_filters.append(x_filter)
            for j in range(self.kernel_size_num - 1):
                x_filter1 = dir_x_filters[j]
                x_filter2 = dir_x_filters[j + 1]
                filter_max = torch.max(x_filter1, x_filter2)
                filter_max = filter_max[:, :, 0:-1:8, 0:-1:8]
                dir_x_filters_max.append(filter_max)
            dir_x_filters_max = torch.cat(dir_x_filters_max, dim=1)
            dir_x_filters_max = dir_x_filters_max.unsqueeze(1)
            x_filters.append(dir_x_filters_max)
        x_filters = torch.cat(x_filters, dim=1)
        x_filters = x_filters.view(x.shape[0], -1)
        return x_filters
