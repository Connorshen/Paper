from experiment.mnist.preprocess.load_data import mnist_loader
from experiment.mnist.preprocess.model import Net
import numpy as np
import torch
import math

theta_num = 16
kernel_size_num = 11
sigma = [2.8, 3.6, 4.5, 5.4, 6.3, 7.3, 8.2, 9.2, 10.2, 11.3, 12.3]
lambd = [3.5, 4.6, 5.6, 6.8, 7.9, 9.1, 10.3, 11.5, 12.7, 14.1, 15.4]
kernel_size = [7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27]
theta = np.arange(0, math.pi, math.pi / theta_num)
# load data
train_loader, test_loader = mnist_loader(batch_size=32,
                                         shuffle=True,
                                         flatten=False,
                                         one_hot=False,
                                         digits=np.arange(0, 10))
# 构建模型
net = Net(kernel_size_num, theta_num, sigma, lambd, kernel_size, theta)
print(net)
for step, (b_img, b_label) in enumerate(train_loader):
    net.train()
    b_output = net(b_img)
    print(1)
