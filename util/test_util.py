# -*- coding: utf-8 -*-
"""
@Time    : 2019/9/5 下午2:12
@Author  : 比尔丶盖子
@Email   : 914138410@qq.com
"""
import torch
from sklearn.metrics import accuracy_score
import numpy as np
from util.data_util import convert_label


def run_testing(net, loss_func, test_loader, gpu=True, digits=np.arange(0, 10)):
    net.eval()
    outputs = []
    labels = []
    predictions = []
    for step, (b_img, b_label) in enumerate(test_loader):
        if torch.cuda.is_available() and gpu:
            b_img = b_img.cuda()
            b_label = b_label.cuda()
        b_label = convert_label(b_label, digits)
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


def run_testing_cluster(net, loss_func, test_loader, gpu=True, digits=np.arange(0, 10)):
    net.eval()
    outputs = []
    labels = []
    predictions = []
    for step, (b_img, b_label) in enumerate(test_loader):
        if torch.cuda.is_available() and gpu:
            b_img = b_img.cuda()
            b_label = b_label.cuda()
        b_label = convert_label(b_label, digits)
        b_output, b_cluster_output = net(b_img)  # 这个地方跟上面那个不一样，多了个输出
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
