# -*- coding: utf-8 -*-
"""
@Time    : 2019/8/13 下午7:56
@Author  : 比尔丶盖子
@Email   : 914138410@qq.com
"""
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

KERNEL_SIZE_NUM = 4
KERNEL_DIRECTION_NUM = 8


def build_filters(kernel_size, sigma, lambd, theta, channel=1):
    filters = []
    filters_origin = []
    for i in range(len(kernel_size)):
        for j in range(len(theta)):
            kernel = cv2.getGaborKernel((kernel_size[i], kernel_size[i]), sigma[i], theta[j], lambd[i], 0.5, 0,
                                        ktype=cv2.CV_32F)

            filters_origin.append(kernel)
            filter_3_temp = tf.expand_dims(kernel, -1)
            filter_3 = filter_3_temp
            for k in range(channel - 1):
                filter_3 = tf.concat([filter_3, filter_3_temp], -1)
            filter_4 = tf.expand_dims(filter_3, -1)
            filters.append(filter_4)
    # show_filters(filters_origin)
    return filters


def show_filters(filters):
    rows = int(len(filters) / KERNEL_DIRECTION_NUM)
    cols = KERNEL_DIRECTION_NUM
    plt.figure(figsize=(10, 10))
    for r_i in range(rows):
        for c_i in range(cols):
            plt.subplot(rows, cols, r_i * cols + c_i + 1)
            f = filters[r_i * cols + c_i]
            f = f.reshape(f.shape[0], f.shape[1])
            plt.imshow(f)
    plt.savefig("convolution_kernel.png", dpi=100)
    plt.show()


gabor_filters = build_filters(kernel_size=np.arange(3, 3 + KERNEL_SIZE_NUM * 2, 2),
                              sigma=np.arange(2, KERNEL_SIZE_NUM + 2),
                              lambd=np.arange(2, KERNEL_SIZE_NUM + 2),
                              theta=np.arange(0, np.pi, np.pi / KERNEL_DIRECTION_NUM))
