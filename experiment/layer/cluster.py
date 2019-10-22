# -*- coding: utf-8 -*-
"""
@Time    : 2019/8/22 下午3:16
@Author  : 比尔丶盖子
@Email   : 914138410@qq.com
"""
from torch import nn
import torch
import torch.nn.functional as F
from torch.nn import Parameter
import scipy.sparse as sparse
import numpy as np


class Cluster(nn.Module):
    def __init__(self, in_features, out_features, n_neuron_cluster, density):
        """
        :param in_features: 输入通道数字
        :param out_features: 输出通道数字
        :param n_neuron_cluster:神经元簇内有几个神经元
        """
        super(Cluster, self).__init__()
        self.n_neuron_cluster = n_neuron_cluster
        self.channel_out = out_features
        self.channel_in = in_features
        self.weight = self.build_sparse_weight(in_features, out_features, density)
        self.weight = Parameter(self.weight, requires_grad=False)
        self.max_index_base = torch.arange(0, self.channel_out, self.n_neuron_cluster)
        self.max_index_base = Parameter(self.max_index_base, requires_grad=False)

    # TODO 目前是按位置group，可改成随机group
    def forward(self, x):  # shape(batch_size,1568)
        # 20000
        n_cluster = int(self.channel_out / self.n_neuron_cluster)
        # shape(batch_size,200000)
        cluster_layer_in = F.linear(x, self.weight)
        # shape(batch_size,20000,10)
        cluster_layer_in_group = cluster_layer_in.view(-1, n_cluster, self.n_neuron_cluster)
        # shape(batch_size,20000)
        max_index_local = torch.argmax(cluster_layer_in_group, dim=2).view(x.size(0), n_cluster)
        max_index_base = self.max_index_base.view(1, n_cluster).repeat(x.size(0), 1)
        max_index = torch.add(max_index_base, max_index_local)
        cluster_layer_out = (cluster_layer_in * 0).scatter(1, max_index, 1)
        return cluster_layer_out

    @staticmethod
    def sparse_to_dense(sp, size):
        row = torch.tensor(sp.row).view(1, -1)
        col = torch.tensor(sp.col).view(1, -1)
        indices = torch.cat([row, col])
        values = torch.tensor(sp.data)
        dense = torch.sparse_coo_tensor(indices=indices,
                                        values=values,
                                        dtype=torch.float,
                                        requires_grad=False,
                                        size=size)
        dense = dense.to_dense()
        return dense

    def build_sparse_weight(self, in_features, out_features, density):
        weight_list = []
        for i in range(in_features):
            weight_one_in_sparse = sparse.rand(out_features, 1, np.random.choice(density), dtype=np.float)
            weight_one_in = self.sparse_to_dense(weight_one_in_sparse, size=(out_features, 1))
            weight_list.append(weight_one_in)
        weight = torch.stack(weight_list, dim=1).view(out_features,in_features)
        return weight
