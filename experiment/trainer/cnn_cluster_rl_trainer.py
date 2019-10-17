from experiment.compare.gabor_cluster_rl.model import Net
import torch
from util.data_util import loader, convert_label
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from experiment.trainer.base_trainer import BaseTrainer


class CnnClusterRlTrainer(BaseTrainer):
    def __init__(self,
                 batch_size,
                 digits,
                 epoch,
                 cluster_layer_weight_density,
                 n_neuron_cluster,
                 n_features_cluster_layer,
                 synaptic_th,
                 learning_rate,
                 use_gpu):
        super().__init__()
        self.batch_size = batch_size
        self.digits = digits
        self.epoch = epoch
        self.use_gpu = use_gpu
        self.n_category = len(digits)
        self.learning_rate = learning_rate
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
                batch_size = len(b_label)
                # forward
                b_output, b_cluster_output = self.net(b_img)  # shape(batch_size,10)
                cluster_weight = self.net.cluster.weight
                loss = self.loss_func(b_output, b_label)
                # backward
                b_predict_prob, b_predict = torch.max(b_output, dim=1)
                b_reward = torch.zeros(batch_size)
                b_reward[b_predict == b_label] = 1
                for i in range(batch_size):
                    reward = b_reward[i]  # 奖励
                    predict_prob = b_predict_prob[i]  # 预测的概率range(0,1)
                    predict = b_predict[i]
                    label = b_label[i]
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
                        modify_weight = modify_weight + self.learning_rate * (reward - predict_prob) * potential
                    else:
                        modify_weight = modify_weight - self.learning_rate * predict_prob * potential
                    modify_weight[modify_weight < 0] = 0
                    weight[predict, :] = modify_weight
                acc = accuracy_score(b_label.data.cpu().numpy(), b_predict.data.cpu().numpy())
                self.loss_all.append(float(loss.data.cpu().numpy()))
                self.acc_all.append(acc)
        self.plot()
        return self.loss_all, self.acc_all
