from experiment.trainer.cnn_bp_trainer import CnnBpTrainer
from experiment.trainer.cnn_cluster_bp_trainer import CnnClusterBpTrainer
from experiment.trainer.cnn_cluster_rl_trainer import CnnClusterRlTrainer
import numpy as np

batch_size = 40
digits = np.array([2, 3, 4])
epoch = 1
cluster_layer_weight_density = 0.01
n_neuron_cluster = 10
n_features_cluster_layer = 50000
learning_rate = 0.1  # 学习率
synaptic_th = 0.8  # 中间层和输出层之间连接矩阵的突触阈值
use_gpu = True
early_stopping_step = 100
valid_interval_step = 2
cnn_bp_trainer = CnnBpTrainer(batch_size,
                              digits,
                              epoch,
                              use_gpu,
                              early_stopping_step,
                              valid_interval_step)
cnn_cluster_bp_trainer = CnnClusterBpTrainer(batch_size,
                                             digits,
                                             epoch,
                                             cluster_layer_weight_density,
                                             n_neuron_cluster,
                                             n_features_cluster_layer,
                                             use_gpu,
                                             early_stopping_step,
                                             valid_interval_step)
cnn_cluster_rl_trainer = CnnClusterRlTrainer(batch_size,
                                             digits,
                                             epoch,
                                             cluster_layer_weight_density,
                                             n_neuron_cluster,
                                             n_features_cluster_layer,
                                             synaptic_th,
                                             learning_rate,
                                             use_gpu,
                                             early_stopping_step,
                                             valid_interval_step)
cnn_bp_trainer.run_training()
cnn_cluster_bp_trainer.run_training()
cnn_cluster_rl_trainer.run_training()
