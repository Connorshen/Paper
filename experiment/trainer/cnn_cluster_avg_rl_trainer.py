from experiment.compare.gabor_cluster_batch_avg_rl.model import Net
import torch
from util.data_util import loader, convert_label
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from util.test_util import run_testing
from experiment.trainer.base_trainer import BaseTrainer


class CnnClusterAvgRlTrainer(BaseTrainer):
    def __init__(self,
                 batch_size,
                 digits,
                 epoch,
                 cluster_layer_weight_density,
                 n_neuron_cluster,
                 n_features_cluster_layer,
                 synaptic_th,
                 use_gpu,
                 early_stopping_step,
                 valid_interval_step):
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
                       n_category=self.n_category,
                       synaptic_th=synaptic_th)
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

    def get_lr(self, b_label, b_cluster_output):
        lr_map = dict()
        for i in range(len(b_label)):
            label = b_label[i]
            if torch.cuda.is_available() and self.use_gpu:
                label = int(label.cpu().numpy())
            else:
                label = int(label.numpy())
            cluster_output = b_cluster_output[i]
            if label in lr_map.keys():
                lr_map[label].append(cluster_output)
            else:
                lr_map[label] = [cluster_output]
        for label in lr_map.keys():
            lr_map[label] = torch.mean(torch.stack(lr_map[label], dim=0), dim=0)
        lr_list = []
        for label in b_label:
            if torch.cuda.is_available() and self.use_gpu:
                label = int(label.cpu().numpy())
            else:
                label = int(label.numpy())
            lr_list.append(lr_map[label])
        lr = torch.stack(lr_list, dim=0)
        return lr

    def update_weight(self, batch_size, b_predict, b_predict_prob, b_label, b_cluster_output):
        b_reward = torch.zeros(batch_size)
        b_reward[b_predict == b_label] = 1
        lr = self.get_lr(b_label, b_cluster_output)
        for i in range(batch_size):
            reward = b_reward[i]  # 奖励
            predict_prob = b_predict_prob[i]  # 预测的概率range(0,1)
            predict = b_predict[i]
            cluster_output = b_cluster_output[i]
            weight = self.net.state_dict()["output.weight"]  # shape(10,n_features_cluster__layer)
            modify_weight = weight[predict, :]  # shape(n_features_cluster__layer)
            rand = torch.rand(modify_weight.shape)
            rand = rand.cuda() if torch.cuda.is_available() and self.use_gpu else rand
            # 权重值越大有越大的概率被改变
            need_modify_weight = (rand < modify_weight).float()
            # 中间层值越大改变的值越大
            potential = torch.mul(cluster_output, need_modify_weight)  # shape(n_features_cluster__layer)
            if reward:
                modify_weight = modify_weight + lr[i] * (reward - predict_prob) * potential
            else:
                modify_weight = modify_weight - lr[i] * predict_prob * potential
            modify_weight[modify_weight < 0] = 0
            weight[predict, :] = modify_weight

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
                batch_size = len(b_label)
                # forward
                b_output, b_cluster_output = self.net(b_img)  # shape(batch_size,10)
                b_predict_prob, b_predict = torch.max(b_output, dim=1)
                # backward
                self.update_weight(batch_size, b_predict, b_predict_prob, b_label, b_cluster_output)
                if total_step % self.valid_interval_step == 0:
                    loss, acc = run_testing(self.net,
                                            self.loss_func,
                                            self.test_loader,
                                            self.use_gpu,
                                            self.digits,
                                            is_rl=True,
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
