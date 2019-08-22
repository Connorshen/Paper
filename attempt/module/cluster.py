# -*- coding: utf-8 -*-
"""
@Time    : 2019/8/22 下午3:16
@Author  : 比尔丶盖子
@Email   : 914138410@qq.com
"""
from torch import nn
import torch


class Cluster(nn.Module):
    def __init__(self, channel_in, channel_out, n_neuron_cluster):
        """
        :param channel_in: 输入通道数字
        :param channel_out: 输出通道数字
        :param n_neuron_cluster:神经元簇内有几个神经元
        """
        super(Cluster, self).__init__()
        self.n_neuron_cluster = n_neuron_cluster
        self.channel_out = channel_out
        self.channel_in = channel_in
        self.weight = torch.randn(channel_in, channel_out, requires_grad=False).cuda()

    def forward(self, x):  # shape(batch_size,1568)
        n_cluster = int(self.channel_out / self.n_neuron_cluster)  # 20000
        cluster_layer_in = torch.mm(x, self.weight)  # shape(batch_size,200000)
        cluster_layer_in_group = cluster_layer_in.view(-1, n_cluster,
                                                       self.n_neuron_cluster)  # shape(batch_size,20000,10)
        max_index_local = torch.argmax(cluster_layer_in_group, dim=2).view(x.size(0),
                                                                           n_cluster).cuda()  # shape(batch_size,20000)
        max_index_base = torch.arange(0, self.channel_out, self.n_neuron_cluster).view(1, n_cluster).repeat(x.size(0),
                                                                                                            1).cuda()
        max_index = torch.add(max_index_base, max_index_local).cuda()
        cluster_layer_out = torch.zeros(cluster_layer_in.shape).cuda().scatter(1, max_index, 1)
        return cluster_layer_out
