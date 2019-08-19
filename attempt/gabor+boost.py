# -*- coding: utf-8 -*-
"""
@Time    : 2019/8/19 下午4:55
@Author  : 比尔丶盖子
@Email   : 914138410@qq.com
"""
import xgboost as xgb
from util.mnist import load_preprocess
from sklearn.metrics import accuracy_score

"""
acc = 96.38%
"""
train_image, train_label, test_image, test_label = load_preprocess(one_hot=False)
train = xgb.DMatrix(train_image, train_label)
test = xgb.DMatrix(test_image, test_label)
params = {
    'booster': 'gbtree',
    # 这里手写数字是0-9，是一个多类的问题，因此采用了multisoft多分类器，
    'objective': 'multi:softmax',
    'num_class': 10,  # 类数，与 multisoftmax 并用
    'gamma': 0.05,  # 在树的叶子节点下一个分区的最小损失，越大算法模型越保守 。[0:]
    'max_depth': 12,  # 构建树的深度 [1:]
    # 'lambda':450,  # L2 正则项权重
    'subsample': 0.4,  # 采样训练数据，设置为0.5，随机选择一般的数据实例 (0:1]
    'colsample_bytree': 0.7,  # 构建树树时的采样比率 (0:1]
    # 'min_child_weight':12, # 节点的最少特征数
    'silent': 1,
    'eta': 0.05,  # 如同学习率
    'seed': 710,
    'nthread': 4,  # cpu 线程数,根据自己U的个数适当调整
}
model = xgb.train(params, train, num_boost_round=10)
y_pred = model.predict(test)
print(accuracy_score(test_label, y_pred))
