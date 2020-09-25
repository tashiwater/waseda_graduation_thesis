#!/usr/bin/env python
# coding:utf-8

import torch
import numpy as np

tau = 5
a = torch.tensor(np.array([[2, 4, 3], [4, 8, 6]]), dtype=torch.float)
b = torch.tensor(np.array([[1, 2, 3], [4, 5, 6]]), dtype=torch.float)
t = 1 - 1.0 / tau
c = t * b + a / tau
print(t)
print(c)
