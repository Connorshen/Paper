from experiment.compare.gabor_bp.model import Net
import torch
from util.data_util import loader, convert_label
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np


class CnnBpTrainer:
    def __init__(self,
                 batch_size,
                 digits,
                 epoch,
                 use_gpu):
        self.batch_size = batch_size
        self.digits = digits
        self.epoch = epoch
        self.use_gpu = use_gpu
        self.n_category = len(digits)
        self.net = Net(n_category=self.n_category)
        if torch.cuda.is_available() and use_gpu:
            self.net = self.net.cuda()
        self.train_loader, self.test_loader = loader(batch_size=batch_size,
                                                     shuffle=True,
                                                     flatten=False,
                                                     one_hot=False,
                                                     digits=digits)
        # 损失函数
        self.loss_func = CrossEntropyLoss()
        # 优化器
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        self.loss_all = []
        self.acc_all = []

    def run_training(self):
        self.loss_all = []
        self.acc_all = []
        print(self.net)
        for e in tqdm(range(self.epoch)):
            for step, (b_img, b_label) in enumerate(self.train_loader):
                self.net.train()
                if torch.cuda.is_available() and self.use_gpu:
                    b_img = b_img.cuda()
                    b_label = b_label.cuda()
                b_label = convert_label(b_label, self.digits)
                b_output = self.net(b_img)
                b_predict = torch.argmax(b_output, dim=1)
                loss = self.loss_func(b_output, b_label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                acc = accuracy_score(b_label.data.cpu().numpy(), b_predict.data.cpu().numpy())
                self.loss_all.append(float(loss.data.cpu().numpy()))
                self.acc_all.append(acc)
        self.plot()
        return self.loss_all, self.acc_all

    def plot(self):
        plt.plot(np.arange(len(self.acc_all)), self.acc_all)
        plt.xlabel("step")
        plt.ylabel("acc")
        plt.title(self.net.name)
        plt.show()
