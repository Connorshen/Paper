from torch.nn import Linear, ReLU, Module, Sequential
import numpy as np
import torch
from origin.data import loader
from util.test_util import run_testing
from torch.nn import CrossEntropyLoss
from util.data_util import convert_label

batch_size = 40
digits = np.arange(10)
n_category = len(digits)
epoch = 10
use_gpu = True
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)


class Net(Module):

    def __init__(self, n_category):
        super(Net, self).__init__()
        self.dense1 = Sequential(Linear(in_features=2560, out_features=100),
                                 ReLU())
        self.dense2 = Sequential(Linear(in_features=100, out_features=n_category))

    def forward(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return x


# 构建模型
net = Net(n_category=n_category)
if torch.cuda.is_available() and use_gpu:
    net = net.cuda()
print(net)
# 数据集
train_loader, test_loader = loader(batch_size=batch_size,
                                   shuffle=True)
# 损失函数
loss_func = CrossEntropyLoss()
# 优化器
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
for e in range(epoch):
    for step, (b_img, b_label) in enumerate(train_loader):
        net.train()
        if torch.cuda.is_available() and use_gpu:
            b_img = b_img.cuda()
            b_label = b_label.cuda()
        b_label = convert_label(b_label, digits)
        b_output = net(b_img)
        loss = loss_func(b_output, b_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            loss, accuracy = run_testing(net, loss_func, test_loader, use_gpu, digits)
            print('Epoch: ', e, '| train loss: %.4f' % loss, '| test accuracy: %.4f' % accuracy)
# 保存模型
