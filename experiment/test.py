# -*- coding: utf-8 -*-
"""
@Time    : 2019/8/22 下午4:20
@Author  : 比尔丶盖子
@Email   : 914138410@qq.com
"""
import torch

a = torch.rand(2000)
b = torch.rand(10, 2000)
c = a < b
print(c.shape)
