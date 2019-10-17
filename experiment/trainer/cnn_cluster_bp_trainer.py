from experiment.compare.gabor_cluster_bp.model import Net
from util.data_util import loader, convert_label
from torch.nn import CrossEntropyLoss
import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from experiment.trainer.base_trainer import BaseTrainer


class CnnClusterBpTrainer(BaseTrainer):
    def __init__(self,
                 batch_size,
                 digits,
                 epoch,
                 cluster_layer_weight_density,
                 n_neuron_cluster,
                 n_features_cluster_layer,
                 use_gpu):
        super().__init__()
        self.batch_size = batch_size
        self.digits = digits
        self.epoch = epoch
        self.use_gpu = use_gpu
        self.n_category = len(digits)
        self.net = Net(n_features_cluster_layer=n_features_cluster_layer,
                       n_neuron_cluster=n_neuron_cluster,
                       cluster_layer_weight_density=cluster_layer_weight_density,
                       n_category=self.n_category)
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
