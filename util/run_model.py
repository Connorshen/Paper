# -*- coding: utf-8 -*-
"""
@Time    : 2019/8/21 下午4:29
@Author  : 比尔丶盖子
@Email   : 914138410@qq.com
"""
import torch
from sklearn.metrics import accuracy_score


def run_testing(net, loss_func, test_loader):
    net.eval()
    outputs = []
    labels = []
    predictions = []
    for step, (b_img, b_label) in enumerate(test_loader):
        if torch.cuda.is_available():
            b_img = b_img.cuda()
            b_label = b_label.cuda()
        b_output = net(b_img)
        b_predict = torch.argmax(b_output, dim=1)
        outputs.append(b_output)
        labels.append(b_label)
        predictions.append(b_predict)
    outputs = torch.cat(outputs, dim=0)
    labels = torch.cat(labels, dim=0)
    predictions = torch.cat(predictions, dim=0)
    loss = loss_func(outputs, labels)
    loss = loss.data.cpu().numpy()
    accuracy = accuracy_score(labels.data.cpu().numpy(), predictions.data.cpu().numpy())
    return loss, accuracy


def run_training(epoch, train_loader, test_loader, net, loss_func, optimizer):
    for e in range(epoch):
        for step, (b_img, b_label) in enumerate(train_loader):
            net.train()
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
                print('Epoch: ', e, '| train loss: %.4f' % loss, '| test accuracy: %.4f' % accuracy)
