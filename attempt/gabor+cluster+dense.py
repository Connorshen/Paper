# -*- coding: utf-8 -*-
"""
@Time    : 2019/8/19 下午7:45
@Author  : 比尔丶盖子
@Email   : 914138410@qq.com
"""
import tensorflow as tf
import numpy as np
from util.mnist import load_preprocess
from tqdm import tqdm


# TODO 速度太慢

class ClusterLayer(tf.keras.layers.Layer):
    def compute_output_signature(self, input_signature):
        pass

    def __init__(self, n_neuron_output, n_neuron_cluster):
        """
        :param n_neuron_output: 有几个神经元输出
        :param n_neuron_cluster: 每个簇包含几个神经元
        """
        super(ClusterLayer, self).__init__()
        self.n_neuron_output = n_neuron_output
        self.n_neuron_cluster = n_neuron_cluster

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel", shape=[self.n_neuron_output, int(input_shape[0])])

    def call(self, input):
        cluster_layer_in = tf.matmul(self.kernel, input)
        cluster_layer_out = None
        cluster_indexes = np.arange(0, self.n_neuron_output, self.n_neuron_cluster)
        for cluster_index in cluster_indexes:
            cluster = cluster_layer_in[cluster_index:cluster_index + self.n_neuron_cluster, :]
            max_index = tf.argmax(cluster)
            cluster_out = tf.transpose(tf.one_hot(max_index, 10))
            if cluster_layer_out is None:
                cluster_layer_out = cluster_out
            else:
                cluster_layer_out = tf.concat([cluster_layer_out, cluster_out], axis=0)
        return cluster_layer_out


tf.random.set_seed(1)
model = tf.keras.Sequential([tf.keras.layers.Dense(128, activation='relu'),
                             tf.keras.layers.Dropout(0.2),
                             tf.keras.layers.Dense(10, activation='softmax')])
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy])

train_image, train_label, test_image, test_label = load_preprocess()
train_image_cluster = []
test_image_cluster = []
cluster_layer = ClusterLayer(10000, 10)
for image in tqdm(train_image):
    train_image_cluster.append(cluster_layer(image.reshape(-1, 1)))
for image in tqdm(test_image):
    test_image_cluster.append(cluster_layer(image.reshape(-1, 1)))
model.fit(train_image_cluster, train_label, epochs=5, batch_size=32)
test_loss, test_acc = model.evaluate(test_image_cluster, test_label)
print(test_acc)
