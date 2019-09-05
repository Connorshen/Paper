# -*- coding: utf-8 -*-
"""
@Time    : 2019/8/22 下午3:15
@Author  : 比尔丶盖子
@Email   : 914138410@qq.com
"""
import torch
from util.data_util import loader
from attempt.layer.gabor import Gabor2d
from attempt.layer.cluster import Cluster
from attempt.layer.output import Output
from util.run_model import run_testing_cluster as run_testing
from util.data_util import convert_label
import numpy as np

EPOCH = 5
BATCH_SIZE = 32  # 32
CLUSTER_LAYER_WEIGHT_DENSITY = 0.001  # 输入层和中间层之间连接矩阵的稀疏度
N_NEURON_CLUSTER = 10  # 每个簇内神经元个数
N_FEATURES_CLUSTER_LAYER = 5000  # 中间层输出神经元个数
LR = 0.1  # 学习率
SYNAPTIC_TH = 0.8  # 中间层和输出层之间连接矩阵的突触阈值
DIGITS = np.array([3, 8])  # 训练的数字:np.array([3,8])
CATEGORY = len(DIGITS)  # 有几类数字
USE_GPU = False  # 是否启用GPU加速
torch.manual_seed(1)
np.random.seed(1)
np.set_printoptions(precision=3, suppress=True)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(  # input shape (1, 28, 28)
            Gabor2d(
                channel_in=1,  # input height
                theta_num=8,  # n directions
                param_num=2,  # n params
                kernel_size=5,  # kernel size
                stride=1,
                padding=2,
            ),  # output shape (16, 28, 28)
            torch.nn.MaxPool2d(kernel_size=2),  # choose max value in 2x2 area, output shape (16, 14, 14)
            torch.nn.ReLU(),  # activation
            torch.nn.BatchNorm2d(16)
        )
        self.conv2 = torch.nn.Sequential(  # input shape (16, 14, 14)
            Gabor2d(16, 8, 4, 5, 1, 2),  # output shape (32, 14, 14)
            torch.nn.ReLU(),  # activation
            torch.nn.MaxPool2d(2),  # output shape (32, 7, 7)
            torch.nn.BatchNorm2d(32),
            torch.nn.Sigmoid()  # 映射到[0,1]
        )
        self.cluster = Cluster(32 * 7 * 7, N_FEATURES_CLUSTER_LAYER, N_NEURON_CLUSTER, CLUSTER_LAYER_WEIGHT_DENSITY)
        self.output = Output(N_FEATURES_CLUSTER_LAYER, CATEGORY, SYNAPTIC_TH)
        self.prob = torch.nn.Sequential(torch.nn.BatchNorm1d(CATEGORY),
                                        torch.nn.Softmax(dim=1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        cluster_out = self.cluster(x)
        x = self.output(cluster_out)
        x = self.prob(x)
        return x, cluster_out


def print_gpu_tensor(tensor, name="", debug=False):
    if debug:
        print(name, tensor.cpu().data.numpy())


net = Net()
if torch.cuda.is_available() and USE_GPU:
    net = net.cuda()
print(net)
loss_func = torch.nn.CrossEntropyLoss()  # loss只用来算准确度
# 数据集
train_loader, test_loader = loader(batch_size=BATCH_SIZE, shuffle=True, flatten=False, one_hot=False, digits=DIGITS)

for e in range(EPOCH):
    for step, (b_img, b_label) in enumerate(train_loader):
        net.train()
        if torch.cuda.is_available() and USE_GPU:
            b_img = b_img.cuda()
            b_label = b_label.cuda()
        b_label = convert_label(b_label, DIGITS)
        batch_size = len(b_label)
        # forward
        b_output, b_cluster_output = net(b_img)  # shape(batch_size,10)
        cluster_weight = net.cluster.weight
        # backward
        b_predict_prob, b_predict = torch.max(b_output, dim=1)
        b_reward = torch.zeros(batch_size)
        b_reward[b_predict == b_label] = 1
        for i in range(batch_size):
            reward = b_reward[i]
            predict_prob = b_predict_prob[i]
            predict = b_predict[i]
            label = b_label[i]
            cluster_output = b_cluster_output[i]
            cluster = net.cluster
            weight = net.state_dict()["output.weight"]  # shape(10,n_features_cluster__layer)
            modify_weight = weight[predict, :]  # shape(n_features_cluster__layer)
            rand = torch.rand(modify_weight.shape)
            rand = rand.cuda() if torch.cuda.is_available() and USE_GPU else rand
            need_modify_weight = (rand < modify_weight).float()
            potential = torch.mul(cluster_output, need_modify_weight)  # shape(n_features_cluster__layer)
            if reward:
                modify_weight = modify_weight + LR * (reward - predict_prob) * potential
            else:
                modify_weight = modify_weight - LR * predict_prob * potential
            modify_weight[modify_weight < 0] = 0
            weight[predict, :] = modify_weight
        if step % 10 == 0:
            loss, accuracy = run_testing(net, loss_func, test_loader, USE_GPU, DIGITS)
            print('Epoch: ', e, '| train loss: %.4f' % loss, '| test accuracy: %.4f' % accuracy)
