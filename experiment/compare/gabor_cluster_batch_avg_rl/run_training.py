# -*- coding: utf-8 -*-
"""
@Time    : 2019/9/5 下午3:33
@Author  : 比尔丶盖子
@Email   : 914138410@qq.com
"""
from experiment.compare.gabor_cluster_rl.model import Net
import numpy as np
from util.data_util import loader, convert_label
from util.test_util import run_testing_cluster as run_testing
import torch
from torch.nn import CrossEntropyLoss

batch_size = 40
digits = np.array([2, 3, 4])
cluster_layer_weight_density = 0.01
n_neuron_cluster = 10
n_features_cluster_layer = 50000
n_category = len(digits)
synaptic_th = 0.8  # 中间层和输出层之间连接矩阵的突触阈值
epoch = 5
use_gpu = False
# 构建模型
net = Net(n_features_cluster_layer=n_features_cluster_layer,
          n_neuron_cluster=n_neuron_cluster,
          cluster_layer_weight_density=cluster_layer_weight_density,
          n_category=n_category,
          synaptic_th=synaptic_th)
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


def get_lr(b_label, b_cluster_output):
    lr_map = dict()
    for i in range(len(b_label)):
        label = b_label[i]
        label = int(label.numpy())
        cluster_output = b_cluster_output[i]
        if label in lr_map.keys():
            lr_map[label].append(cluster_output)
        else:
            lr_map[label] = [cluster_output]
    for label in lr_map.keys():
        lr_map[label] = torch.mean(torch.stack(lr_map[label], dim=0), dim=0)
    lr_list = []
    for label in b_label:
        label = int(label.numpy())
        lr_list.append(lr_map[label])
    lr = torch.stack(lr_list, dim=0)
    return lr


for e in range(epoch):
    # b开头的变量表示关于batch的变量
    for step, (b_img, b_label) in enumerate(train_loader):
        net.train()
        if torch.cuda.is_available() and use_gpu:
            b_img = b_img.cuda()
            b_label = b_label.cuda()
        b_label = convert_label(b_label, digits)
        batch_size = len(b_label)
        # forward
        b_output, b_cluster_output = net(b_img)  # shape(batch_size,10)
        cluster_weight = net.cluster.weight
        # backward
        b_predict_prob, b_predict = torch.max(b_output, dim=1)
        b_reward = torch.zeros(batch_size)
        b_reward[b_predict == b_label] = 1
        lr = get_lr(b_label, b_cluster_output)
        for i in range(batch_size):
            reward = b_reward[i]  # 奖励
            predict_prob = b_predict_prob[i]  # 预测的概率range(0,1)
            predict = b_predict[i]
            label = b_label[i]
            cluster_output = b_cluster_output[i]
            weight = net.state_dict()["output.weight"]  # shape(10,n_features_cluster__layer)
            modify_weight = weight[predict, :]  # shape(n_features_cluster__layer)
            rand = torch.rand(modify_weight.shape)
            rand = rand.cuda() if torch.cuda.is_available() and use_gpu else rand
            # 权重值越大有越大的概率被改变
            need_modify_weight = (rand < modify_weight).float()
            # 中间层值越大改变的值越大
            potential = torch.mul(cluster_output, need_modify_weight)  # shape(n_features_cluster__layer)
            if reward:
                modify_weight = modify_weight + lr[i] * (reward - predict_prob) * potential
            else:
                modify_weight = modify_weight - lr[i] * predict_prob * potential
            modify_weight[modify_weight < 0] = 0
            weight[predict, :] = modify_weight
        if step % 100 == 0:
            loss, accuracy = run_testing(net, loss_func, test_loader, use_gpu, digits)
            print('Epoch: ', e, '| train loss: %.4f' % loss, '| test accuracy: %.4f' % accuracy)