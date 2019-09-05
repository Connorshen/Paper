# -*- coding: utf-8 -*-
"""
@Time    : 2019/9/5 下午2:38
@Author  : 比尔丶盖子
@Email   : 914138410@qq.com
"""
from experiment.compare.gabor_cluster_bp.model import Net
import numpy as np
from util.data_util import loader
from util.test_util import run_testing
import torch
from torch.nn import CrossEntropyLoss
from experiment.static import config
from os import path

batch_size = 32
digits = np.array([3, 5])
use_gpu = True
# 加载模型
net = torch.load(path.join(config.MODEL_STATE_PATH, Net.name))
if torch.cuda.is_available() and use_gpu:
    net = net.cuda()
# 数据集
train_loader, test_loader = loader(batch_size=batch_size,
                                   shuffle=True,
                                   flatten=False,
                                   one_hot=False,
                                   digits=digits)
# 损失函数
loss_func = CrossEntropyLoss()
# testing
loss, accuracy = run_testing(net, loss_func, test_loader, use_gpu, digits)
print("test accuracy: %.4f" % accuracy)
