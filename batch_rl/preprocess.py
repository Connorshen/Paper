import scipy.io as scio
from torch.nn import BatchNorm2d, ReLU, MaxPool2d, Module, Sequential
from experiment.layer.gabor import Gabor2d
from util.data_util import loader
import numpy as np
from tqdm import tqdm


class Net(Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = Sequential(  # input shape (1, 28, 28)
            Gabor2d(
                channel_in=1,  # input height
                theta_num=8,  # n directions
                param_num=2,  # n params
                kernel_size=5,  # kernel size
                stride=1,
                padding=2,
            ),  # output shape (16, 28, 28)
            MaxPool2d(kernel_size=2),  # choose max value in 2x2 area, output shape (16, 14, 14)
            BatchNorm2d(16),
            ReLU()  # activation
        )
        self.conv2 = Sequential(  # input shape (16, 14, 14)
            Gabor2d(16, 8, 4, 5, 1, 2),  # output shape (32, 14, 14)
            MaxPool2d(2),  # output shape (32, 7, 7)
            BatchNorm2d(32),
            ReLU()  # activation
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        return x


# 构建模型
net = Net()
net = net.cuda()
print(net)
# 数据集
train_loader, test_loader = loader(batch_size=100,
                                   shuffle=True,
                                   flatten=False,
                                   one_hot=False,
                                   digits=np.arange(10))
# 前向
train_img = []
train_label = []
test_img = []
test_label = []
origin_train_img = []
origin_test_img = []
for step, (b_img, b_label) in tqdm(enumerate(train_loader)):
    net.train()
    b_img = b_img.cuda()
    b_label = b_label.cuda()
    b_output = net(b_img)
    b_label = b_label.view(-1, 1)
    train_img.append(b_output.cpu().detach().numpy())
    train_label.append(b_label.cpu().detach().numpy())
    origin_train_img.append(b_img.cpu().detach().numpy())
for step, (b_img, b_label) in tqdm(enumerate(test_loader)):
    net.train()
    b_img = b_img.cuda()
    b_label = b_label.cuda()
    b_output = net(b_img)
    b_label = b_label.view(-1, 1)
    test_img.append(b_output.cpu().detach().numpy())
    test_label.append(b_label.cpu().detach().numpy())
    origin_test_img.append(b_img.cpu().detach().numpy())
train_img = np.vstack(train_img)
train_label = np.vstack(train_label)
origin_train_img = np.vstack(origin_train_img)
test_img = np.vstack(test_img)
test_label = np.vstack(test_label)
origin_test_img = np.vstack(origin_test_img)
scio.savemat("train.mat", {"train_img": train_img, "train_label": train_label, "origin_train_img": origin_train_img})
scio.savemat("test.mat", {"test_img": test_img, "test_label": test_label, "origin_test_img": origin_test_img})
