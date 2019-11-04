import torch
from python.data_util import fashion_loader
from sklearn.metrics import accuracy_score

"""
acc = 88.46%
"""
EPOCH = 1
BATCH_SIZE = 32
LR = 0.001
torch.manual_seed(1)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(  # input shape (1, 28, 28)
            torch.nn.Conv2d(
                in_channels=1,  # input height
                out_channels=16,  # n_filters
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
            torch.nn.Conv2d(16, 32, 5, 1, 2),  # output shape (32, 14, 14)
            torch.nn.ReLU(),  # activation
            torch.nn.MaxPool2d(2),  # output shape (32, 7, 7)
            torch.nn.BatchNorm2d(32)
        )
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(32 * 7 * 7, 128),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
        )
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Linear(64, 10)  # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = self.fc1(x)
        x = self.fc2(x)
        output = self.out(x)
        return output  # return x for visualization


def run_testing(net, loss_func, test_loader, gpu=True):
    net.eval()
    outputs = []
    labels = []
    predictions = []
    for step, (b_img, b_label) in enumerate(test_loader):
        if torch.cuda.is_available() and gpu:
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


def run_training(epoch, train_loader, test_loader, net, loss_func, optimizer, gpu=True):
    for e in range(epoch):
        for step, (b_img, b_label) in enumerate(train_loader):
            net.train()
            if torch.cuda.is_available() and gpu:
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


net = Net()
if torch.cuda.is_available():
    net = net.cuda()
print(net)
# 优化器
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
# 损失函数
loss_func = torch.nn.CrossEntropyLoss()
# 数据集
train_loader, test_loader = fashion_loader(batch_size=BATCH_SIZE, shuffle=True, flatten=False, one_hot=False)
# train
run_training(EPOCH, train_loader, test_loader, net, loss_func, optimizer)
# test
loss, accuracy = run_testing(net, loss_func, test_loader)
print('test accuracy: %.4f' % accuracy)
