# -*- coding: utf-8 -*-
"""
@Time    : 2019/8/14 下午4:24
@Author  : 比尔丶盖子
@Email   : 914138410@qq.com
"""
import torch
from util.mnist import loader
from util.run_model import run_testing, run_training
from attempt.module.gabor_gard import GaborConv2d

"""
acc = 98.58%
"""
EPOCH = 5
BATCH_SIZE = 32
LR = 0.001


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(  # input shape (1, 28, 28)
            GaborConv2d(
                channel_in=1,  # input height
                channel_out=16,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
                # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),  # output shape (16, 28, 28)
            torch.nn.MaxPool2d(kernel_size=2),  # choose max value in 2x2 area, output shape (16, 14, 14)
            torch.nn.ReLU(),  # activation
            torch.nn.BatchNorm2d(16)
        )
        self.conv2 = torch.nn.Sequential(  # input shape (16, 14, 14)
            GaborConv2d(16, 32, 5, 1, 2),  # output shape (32, 14, 14)
            torch.nn.ReLU(),  # activation
            torch.nn.MaxPool2d(2),  # output shape (32, 7, 7)
            torch.nn.BatchNorm2d(32)
        )
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(32 * 7 * 7, 128),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Linear(128, 10)  # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = self.fc1(x)
        output = self.out(x)
        return output  # return x for visualization


net = Net()
if torch.cuda.is_available():
    net = net.cuda()
print(net)
# 优化器
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
# 损失函数
loss_func = torch.nn.CrossEntropyLoss()
# 数据集
train_loader, test_loader = loader(batch_size=BATCH_SIZE, shuffle=True, flatten=False, one_hot=False)
# train
run_training(EPOCH, train_loader, test_loader, net, loss_func, optimizer)
# test
loss, accuracy = run_testing(net, loss_func, test_loader)
print('test accuracy: %.4f' % accuracy)
