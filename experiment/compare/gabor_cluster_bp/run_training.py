# -*- coding: utf-8 -*-
"""
@Time    : 2019/9/5 下午1:57
@Author  : 比尔丶盖子
@Email   : 914138410@qq.com
"""
from experiment.compare.gabor_cluster_bp.model import Net
import numpy as np
from util.data_util import loader, convert_label
from util.test_util import run_testing
import torch
from torch.nn import CrossEntropyLoss
from experiment.static import config
from os import path

batch_size = 1000
digits = np.arange(10)
cluster_layer_weight_density = np.arange(0.05, 0.11, 0.01, dtype=np.float)
n_neuron_cluster = 10
n_features_cluster_layer = 50000
n_category = len(digits)
epoch = 10
use_gpu = False
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
# 构建模型
net = Net(n_features_cluster_layer=n_features_cluster_layer,
          n_neuron_cluster=n_neuron_cluster,
          cluster_layer_weight_density=cluster_layer_weight_density,
          n_category=n_category)
if torch.cuda.is_available() and use_gpu:
    net = net.cuda()
print(net)
# 数据集
train_loader, test_loader = loader(batch_size=batch_size,
                                   shuffle=True,
                                   flatten=False,
                                   one_hot=False,
                                   digits=digits)
# 损失函数
loss_func = CrossEntropyLoss()
# 优化器
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
for e in range(epoch):
    for step, (b_img, b_label) in enumerate(train_loader):
        net.train()
        if torch.cuda.is_available() and use_gpu:
            b_img = b_img.cuda()
            b_label = b_label.cuda()
        b_label = convert_label(b_label, digits)
        b_output = net(b_img)
        loss = loss_func(b_output, b_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            loss, accuracy = run_testing(net, loss_func, test_loader, use_gpu, digits)
            print('Epoch: ', e, '| train loss: %.4f' % loss, '| test accuracy: %.4f' % accuracy)
# 保存模型
torch.save(net, path.join(config.MODEL_STATE_PATH, Net.name))
