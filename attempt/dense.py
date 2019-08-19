# -*- coding: utf-8 -*-
"""
@Time    : 2019-08-12 16:32
@Author  : 比尔丶盖子
@Email   : 914138410@qq.com
"""
import tensorflow as tf
from util.mnist import load_mnist

"""
acc = 96.81%
"""
tf.random.set_seed(1)
model = tf.keras.Sequential([tf.keras.layers.Dense(784, activation='relu'),
                             tf.keras.layers.Dense(128, activation='relu'),
                             tf.keras.layers.Dropout(0.2),
                             tf.keras.layers.Dense(10, activation='softmax')])
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy])

train_image, train_label, test_image, test_label = load_mnist()

model.fit(train_image, train_label, epochs=1)
test_loss, test_acc = model.evaluate(test_image, test_label)
print(test_acc)
