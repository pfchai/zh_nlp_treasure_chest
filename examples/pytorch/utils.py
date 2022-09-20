# -*- coding: utf-8 -*-

import random

import torch
import numpy as np


def set_seed(seed):
    # seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # 并行gpu
        torch.cuda.manual_seed_all(seed)
        # cpu/gpu结果一致
        torch.backends.cudnn.deterministic = True
        # 训练集变化不大时使训练加速
        # torch.backends.cudnn.benchmark = True