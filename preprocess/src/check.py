#!/usr/bin/env python3
# coding: utf-8
import os
from pathlib import Path
import pandas as pd
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = CURRENT_DIR + "/../data/"
dir_path = DATA_DIR + "connect_input/tactile_raw/"
paths = [str(p) for p in Path(dir_path).glob("./*.csv")]
paths.sort()

for i, path in enumerate(paths):
    df = pd.read_csv(path, header=None, index_col=None)
    for index, row in df.iterrows():
        if any(row > 0.1):
            print("{} : {} row".format(i, index / 4))
            break

    # tail = df.iloc[-1]
    # max_val = tail.max()
    # if max_val > 0.1:
    #     all_max = df.max(axis=0).max()
    #     print(i, max_val, all_max)
