# -*- coding: utf-8 -*-
"""
@Time    : 2019/9/5 下午1:39
@Author  : 比尔丶盖子
@Email   : 914138410@qq.com
"""
from torch.nn import Linear, BatchNorm2d, ReLU, MaxPool2d, Module, Sequential
from attempt.layer.gabor import Gabor2d
from attempt.layer.cluster import Cluster


class Net(Module):
    name = "gabor_cluster_dense_net"

    def __init__(self,
                 n_features_cluster_layer,
                 n_neuron_cluster,
                 cluster_layer_weight_density,
                 n_category):
        super(Net, self).__init__()
        self.conv1 = Sequential(  # input shape (1, 28, 28)
            Gabor2d(
                channel_in=1,  # input height
                theta_num=8,  # n directions
                param_num=2,  # n params
                kernel_size=5,  # kernel size
                stride=1,
                padding=2,
            ),  # output shape (16, 28, 28)
            MaxPool2d(kernel_size=2),  # choose max value in 2x2 area, output shape (16, 14, 14)
            ReLU(),  # activation
            BatchNorm2d(16)
        )
        self.conv2 = Sequential(  # input shape (16, 14, 14)
            Gabor2d(16, 8, 4, 5, 1, 2),  # output shape (32, 14, 14)
            ReLU(),  # activation
            MaxPool2d(2),  # output shape (32, 7, 7)
            BatchNorm2d(32)
        )
        self.cluster = Cluster(32 * 7 * 7, n_features_cluster_layer, n_neuron_cluster, cluster_layer_weight_density)
        self.output = Linear(n_features_cluster_layer, n_category)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = self.cluster(x)
        x = self.output(x)
        return x
