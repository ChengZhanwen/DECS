#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :seed.py
# @Time      :2023/4/3 15:15
# @Author    :luojiachen
def set_seed(seed=2333):
    # print("SET SEED is Called ")
    # try:
    #     import torch
    #     torch.manual_seed(seed)
    #     if torch.cuda.is_available():
    #         torch.cuda.manual_seed_all(seed)
    #         torch.backends.cudnn.deterministic = True
    #         torch.backends.cudnn.benchmark = False
    # except Exception as e:
    #     print("Set seed failed,details are ", e)
    #     pass
    # import numpy as np
    # np.random.seed(seed)
    # import random as python_random
    # python_random.seed(seed)

    import random,os, torch, numpy as np
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True