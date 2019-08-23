# -*- coding: utf-8 -*-
"""
@Time    : 2019/8/22 下午4:20
@Author  : 比尔丶盖子
@Email   : 914138410@qq.com
"""
import torch

print
torch.sparse.FloatTensor(2, 3)
# 输入如下内容
# FloatTensor of size 2x3 with indices:
# [torch.LongTensor with no dimension]
# and values:
# [torch.FloatTensor with no dimension]

# 我们输出tenser中的数组
print(torch.sparse.FloatTensor(2, 3).to_dense())
# 输出结果类似于
print(torch.zeros(2, 3))
