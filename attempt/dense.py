# -*- coding: utf-8 -*-
"""
@Time    : 2019-08-12 16:32
@Author  : 比尔丶盖子
@Email   : 914138410@qq.com
"""

import torch
from util.mnist import loader
from util.model_test import run_testing

"""
acc = 96.96%
"""
EPOCH = 5
BATCH_SIZE = 32
LR = 0.001


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.dense = torch.nn.Sequential(
            torch.nn.Linear(784, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 10),
            torch.nn.Softmax(1)
        )

    def forward(self, x):
        output = self.dense(x)
        return output


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
net.train()
for epoch in range(EPOCH):
    for step, (b_img, b_label) in enumerate(train_loader):
        if torch.cuda.is_available():
            b_img = b_img.cuda()
            b_label = b_label.cuda()
        b_output = net(b_img)
        loss = loss_func(b_output, b_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            loss, accuracy = run_testing(net, loss_func, test_loader)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss, '| test accuracy: %.4f' % accuracy)
# test
net.eval()
loss, accuracy = run_testing(net, loss_func, test_loader)
print('test accuracy: %.4f' % accuracy)
