# -*- coding: utf-8 -*-
"""
@Time    : 2019-08-13 09:18
@Author  : 比尔丶盖子
@Email   : 914138410@qq.com
"""

import tensorflow as tf
from util.mnist import load_mnist

"""
acc = 99.26%
"""
tf.random.set_seed(1)
model = tf.keras.Sequential([tf.keras.layers.Conv2D(32, (5, 5), activation='relu'),
                             tf.keras.layers.MaxPooling2D((2, 2)),
                             tf.keras.layers.Conv2D(64, (5, 5), activation='relu'),
                             tf.keras.layers.MaxPooling2D((2, 2)),
                             tf.keras.layers.Flatten(),
                             tf.keras.layers.Dropout(0.2),
                             tf.keras.layers.Dense(128, activation='relu'),
                             tf.keras.layers.Dropout(0.2),
                             tf.keras.layers.Dense(10, activation='softmax')])

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy])
train_image, train_label, test_image, test_label = load_mnist()
train_image = train_image.reshape(-1, 28, 28, 1)
test_image = test_image.reshape(-1, 28, 28, 1)

model.fit(train_image, train_label, epochs=5)
model.summary()
test_loss, test_acc = model.evaluate(test_image, test_label)
print(test_acc)
