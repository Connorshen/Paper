# -*- coding: utf-8 -*-
"""
@Time    : 2019/9/5 下午2:26
@Author  : 比尔丶盖子
@Email   : 914138410@qq.com
"""
import os


class Config:
    ROOT = None
    MODEL_STATE_PATH = None

    def __init__(self):
        self.ROOT = os.path.dirname(__file__)
        self.MODEL_STATE_PATH = os.path.join(self.ROOT, "model_state")
        self.verify_path()

    def verify_path(self):
        if os.path.exists(self.MODEL_STATE_PATH) is not True:
            os.mkdir(self.MODEL_STATE_PATH)
