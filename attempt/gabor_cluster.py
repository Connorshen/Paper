# -*- coding: utf-8 -*-
"""
@Time    : 2019/8/22 下午3:15
@Author  : 比尔丶盖子
@Email   : 914138410@qq.com
"""
import torch
from util.mnist import loader
from attempt.layer.gabor import Gabor2d
from attempt.layer.cluster import Cluster
from attempt.layer.output import Output
from util.run_model import run_testing

EPOCH = 5
BATCH_SIZE = 32
LR = 0.001
torch.manual_seed(1)


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
            torch.nn.BatchNorm2d(32)
        )
        self.cluster = Cluster(in_features=32 * 7 * 7, out_features=200000, n_neuron_cluster=10)
        self.output = Output(in_features=200000, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = self.cluster(x)
        x = self.output(x)
        return x


net = Net()
if torch.cuda.is_available():
    net = net.cuda()
print(net)
loss_func = torch.nn.CrossEntropyLoss()
# 数据集
train_loader, test_loader = loader(batch_size=BATCH_SIZE, shuffle=True, flatten=False, one_hot=False)
# forward
for e in range(EPOCH):
    for step, (b_img, b_label) in enumerate(train_loader):
        net.train()
        if torch.cuda.is_available():
            b_img = b_img.cuda()
            b_label = b_label.cuda()
        b_output = net(b_img)
        loss, accuracy = run_testing(net, loss_func, test_loader)
        print(loss)
        print(accuracy)
        break
    break
