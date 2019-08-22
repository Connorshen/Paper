# -*- coding: utf-8 -*-
"""
@Time    : 2019/8/22 下午4:20
@Author  : 比尔丶盖子
@Email   : 914138410@qq.com
"""
import torch

a = torch.arange(15, dtype=torch.float).view(5, 3)
b = torch.mean(a, 1)
print(a)
print(b)
