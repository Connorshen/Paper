from experiment.compare.gabor_cluster_batch_difference_rl.model import Net
import torch
from util.data_util import loader, convert_label
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from util.test_util import run_testing
from experiment.trainer.base_trainer import BaseTrainer


class CnnClusterDiffRlTrainer(BaseTrainer):
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

    def get_lr(self, batch_size, labels, cluster_outputs):
        cluster_out_map = dict()
        for lb in labels:
            if torch.cuda.is_available() and self.use_gpu:
                lb = int(lb.cpu().numpy())
            else:
                lb = int(lb.numpy())
            cluster_out_map[lb] = {"self_percent": [],
                                   "other_percent": [],
                                   "cluster_out": [],
                                   "cluster_out_sum": [],
                                   "lr": 0,
                                   "self_num": 0,
                                   "other_num": 0}
        for index in range(len(labels)):
            lb = labels[index]
            if torch.cuda.is_available() and self.use_gpu:
                lb = int(lb.cpu().numpy())
            else:
                lb = int(lb.numpy())
            out = cluster_outputs[index]
            cluster_out_map[lb]["cluster_out"].append(out)
        for lb in cluster_out_map.keys():
            cluster_out_map[lb]["self_num"] = len(cluster_out_map[lb]["cluster_out"])
            cluster_out_map[lb]["other_num"] = batch_size - cluster_out_map[lb]["self_num"]
            cluster_out_map[lb]["cluster_out"] = torch.stack(cluster_out_map[lb]["cluster_out"], dim=0)
            cluster_out_map[lb]["cluster_out_sum"] = torch.sum(cluster_out_map[lb]["cluster_out"], dim=0)
            cluster_out_map[lb]["self_percent"] = cluster_out_map[lb]["cluster_out_sum"] / cluster_out_map[lb][
                "self_num"]
        for lb in cluster_out_map.keys():
            other_out_sum_list = []
            for other_lb in cluster_out_map.keys():
                if other_lb != lb:
                    other_out_sum_list.append(cluster_out_map[other_lb]["cluster_out_sum"])
            other_out_sum = torch.sum(torch.stack(other_out_sum_list, dim=0), dim=0)
            cluster_out_map[lb]["other_percent"] = other_out_sum / cluster_out_map[lb]["other_num"]
            diff = cluster_out_map[lb]["self_percent"] - cluster_out_map[lb]["other_percent"]
            cluster_out_map[lb]["lr"] = 1 / (1 + torch.exp(-diff))
        lr_list = []
        for lb in labels:
            if torch.cuda.is_available() and self.use_gpu:
                lb = int(lb.cpu().numpy())
            else:
                lb = int(lb.numpy())
            lr_list.append(cluster_out_map[lb]["lr"])
        return torch.stack(lr_list, dim=0)

    def update_weight(self, batch_size, b_predict, b_predict_prob, b_label, b_cluster_output):
        b_reward = torch.zeros(batch_size)
        b_reward[b_predict == b_label] = 1
        lr = self.get_lr(batch_size, b_label, b_cluster_output)
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
