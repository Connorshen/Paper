from experiment.compare.gabor_cluster_bp.model import Net
from util.data_util import loader, convert_label
from torch.nn import CrossEntropyLoss
import torch
from util.test_util import run_testing
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
                 use_gpu,
                 early_stopping_step=None,
                 valid_interval_step=None):
        super().__init__()
        self.batch_size = batch_size
        self.digits = digits
        self.epoch = epoch
        self.use_gpu = use_gpu
        self.n_category = len(digits)
        self.early_stopping_step = early_stopping_step
        self.valid_interval_step = valid_interval_step
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
        self.step_all = []
        total_step = 0
        print(self.net)
        for e in tqdm(range(self.epoch)):
            for step, (b_img, b_label) in enumerate(self.train_loader):
                self.net.train()
                if torch.cuda.is_available() and self.use_gpu:
                    b_img = b_img.cuda()
                    b_label = b_label.cuda()
                b_label = convert_label(b_label, self.digits)
                b_output = self.net(b_img)
                loss = self.loss_func(b_output, b_label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if total_step % self.valid_interval_step == 0:
                    loss, acc = run_testing(self.net,
                                            self.loss_func,
                                            self.test_loader,
                                            self.use_gpu,
                                            self.digits,
                                            is_rl=False,
                                            break_step=100)
                    print('Epoch: ', e, '| train loss: %.4f' % loss, '| test accuracy: %.4f' % acc)
                    self.loss_all.append(loss)
                    self.acc_all.append(acc)
                    self.step_all.append(total_step)
                if total_step == self.early_stopping_step:
                    break
                total_step += 1
        self.plot()
        return self.loss_all, self.acc_all
