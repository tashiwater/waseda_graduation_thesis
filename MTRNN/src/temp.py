#!/usr/bin/env python
# coding:utf-8

import torch
import numpy as np


a = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
a1 = np.array([[[1, 2, 3]], [[7, 8, 9]]])
print(a1)
for i in range(3):
    a = np.concatenate([a1, a], 1)
print(a)
print(a.shape)
print(a)
