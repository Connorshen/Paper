# -*- coding: utf-8 -*-
"""
@Time    : 2019/8/22 上午10:56
@Author  : 比尔丶盖子
@Email   : 914138410@qq.com
"""
import torch.nn.functional as F
import torch
from torch import nn
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
Gabor固定
"""


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


def gabor_filter(kernel_size, channel_in, theta_num, param_num):
    """
       .   @param ksize Size of the filter returned.
       .   @param sigma Standard deviation of the gaussian envelope.可以看出，随着σ的增大，条纹数量越多
       .   @param theta Orientation of the normal to the parallel stripes of a Gabor function.角度要转换为弧度
       .   @param lambd Wavelength of the sinusoidal factor.波长越大，黑白相间的间隔越大
       .   @param gamma Spatial aspect ratio.随着γ的增大，核函数图像形状会发生改变，γ越小，核函数图像会越高，随着其增大，图像会变的越矮。
       .   @param psi Phase offset.当ψ为0时以白条为中心，当ψ为180时，以黑条为中心
       .   @param ktype Type of filter coefficients. It can be CV_32F or CV_64F .
       """
    channel_out = theta_num * param_num
    theta = np.arange(0, math.pi, math.pi / theta_num)
    sigma = np.arange(1, 1 + param_num)
    lambd = np.arange(1, 1 + param_num)
    kernels = []
    for i in range(param_num):
        for j in range(theta_num):
            kernel = cv2.getGaborKernel(ksize=(kernel_size, kernel_size), sigma=sigma[i], theta=theta[j],
                                        lambd=lambd[i], gamma=0.5, psi=0, ktype=cv2.CV_32F)
            kernel = torch.tensor(kernel).cuda()
            kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(1, channel_in, 1, 1)
            kernels.append(kernel)

    kernels = torch.cat(kernels, dim=0)
    return kernels


class Gabor2d(nn.Module):
    def __init__(self, channel_in, theta_num, param_num, kernel_size, stride=1, padding=0):
        super(Gabor2d, self).__init__()
        self.stride = stride
        self.padding = padding
        self.kernels = gabor_filter(kernel_size=kernel_size, channel_in=channel_in, theta_num=theta_num,
                                    param_num=param_num)  # [channel_out, channel_in, kernel, kernel]

    def forward(self, x):
        out = F.conv2d(x, self.kernels, stride=self.stride, padding=self.padding)
        return out
