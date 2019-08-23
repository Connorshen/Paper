# -*- coding: utf-8 -*-
"""
@Time    : 2019-08-12 16:32
@Author  : 比尔丶盖子
@Email   : 914138410@qq.com
"""

import torch
from util.mnist import loader
from util.run_model import run_testing, run_training

"""
acc = 97.51%
"""
EPOCH = 5
BATCH_SIZE = 32
LR = 0.001
torch.manual_seed(1)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(784, 128),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
        )
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
        )
        self.fc3 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


net = Net()
if torch.cuda.is_available():
    net = net.cuda()
print(net)
# 优化器
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
# 损失函数
loss_func = torch.nn.CrossEntropyLoss()
# 数据集
train_loader, test_loader = loader(batch_size=BATCH_SIZE, shuffle=True)
# train
run_training(EPOCH, train_loader, test_loader, net, loss_func, optimizer)
# test
loss, accuracy = run_testing(net, loss_func, test_loader)
print('test accuracy: %.4f' % accuracy)
