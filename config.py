# -*- coding: utf-8 -*-
# @author: LI Qiang
# @email: qiangli@link.cuhk.edu.hk 
# @date: 2023/01/26

import os

class AlgoConfig(object):
    """定义一个关于算法的默认设置类"""

    def __init__(self):
        self.log_scale = True
        self.num_points = 2000 # 如果使用对数采样，图中的点数是self.num_points

        self.max_iter_log = 7
        self.max_iter_num = 10 ** int(self.max_iter_log)
        self.step = 300 # 隔self.step采样一次
        self.batch = 1

config = AlgoConfig()