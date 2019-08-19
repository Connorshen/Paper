# -*- coding: utf-8 -*-
"""
@Time    : 2019/8/13 下午8:03
@Author  : 比尔丶盖子
@Email   : 914138410@qq.com
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from util.mnist import load_mnist
from preprocess.filter import gabor_filters
from url import DataConfig

"""gabor滤波处理后并保存特征向量"""
BATCH_NUM = 10000  # 防止内存溢出


@tf.function
def gabor_process(data, filters):
    feature_map = []
    for flt in filters:
        convolution_map = tf.nn.conv2d(data, flt, strides=[1, 1, 1, 1], padding='SAME')
        max_pool_map = tf.nn.max_pool(convolution_map, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
        feature_map.append(max_pool_map)
    feature_map = tf.concat(feature_map, -1)
    feature_map = tf.reshape(feature_map, (-1, 7 * 7 * 32))
    return feature_map


def process(data):
    result_feature_map = []
    for i in tqdm(range(0, len(data), BATCH_NUM)):
        feature_map = gabor_process(data[i:i + BATCH_NUM], gabor_filters)
        result_feature_map.append(feature_map)
    result_feature_map = tf.concat(result_feature_map, 0)
    result_feature_map = result_feature_map.numpy()
    return result_feature_map


train_image, train_label, test_image, test_label = load_mnist(flatten=False, one_hot=False)

train_feature_map = process(train_image)
train = pd.DataFrame(train_feature_map)
train["label"] = train_label
train.to_csv(DataConfig.PREPROCESS_TRAIN, index=False)

test_feature_map = process(test_image)
test = pd.DataFrame(test_feature_map)
test["label"] = test_label
test.to_csv(DataConfig.PREPROCESS_TEST, index=False)
