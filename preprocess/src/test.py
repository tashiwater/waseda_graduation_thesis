#!/usr/bin/env python3
# coding: utf-8
import numpy as np

a = np.array([[0, 1], [2, 3], [4, 5], [6, 7]])
print(a.reshape(2, -1))
