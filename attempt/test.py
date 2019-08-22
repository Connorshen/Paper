# -*- coding: utf-8 -*-
"""
@Time    : 2019/8/22 下午4:20
@Author  : 比尔丶盖子
@Email   : 914138410@qq.com
"""
import torch

torch.manual_seed(1)
class_num = 10
batch_size = 4
label = torch.LongTensor(batch_size, 1).random_() % class_num
label = torch.cat([label, label - 1], 1)
one_hot = torch.zeros(batch_size, class_num).scatter_(1, label, 1)
print(label)
print(one_hot)
