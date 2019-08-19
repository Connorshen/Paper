# -*- coding: utf-8 -*-
"""
@Time    : 2019/8/14 下午4:24
@Author  : 比尔丶盖子
@Email   : 914138410@qq.com
"""
import tensorflow as tf
from util.mnist import load_preprocess

"""
acc = 98.34%
"""
tf.random.set_seed(1)
model = tf.keras.Sequential([tf.keras.layers.Dense(100, activation='relu'),
                             tf.keras.layers.Dropout(0.2),
                             tf.keras.layers.Dense(10, activation='softmax')])
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy])

train_image, train_label, test_image, test_label = load_preprocess()

model.fit(train_image, train_label, epochs=5, batch_size=32)
test_loss, test_acc = model.evaluate(test_image, test_label)
print(test_acc)
