#!/usr/bin/env python3
# coding: utf-8
import pandas as pd
import os
from pathlib import Path
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = CURRENT_DIR + "/../data/"
dir_path = DATA_DIR + "connect_input/image_feature/"
paths = [str(p) for p in Path(dir_path).glob("./*.csv")]
paths.sort()
# dfs = [pd.read_csv(path, header=None).iloc[:, :] for path in paths]
dfs = [pd.read_csv(path, header=None).iloc[:, :] for path in paths]
df = pd.concat(dfs)

pd.set_option("display.max_columns", 100)
print(df.describe())
for mi, ma in zip(df.min(), df.max()):
    print("[{}, {}],".format(mi, ma))


# for d in dfs:
#     print(d.head(1))
# for i, path in enumerate(paths):
#     df = pd.read_csv(path)

#     tail = df.iloc[-1]
#     max_val = tail.max()
#     if max_val > 0.1:
#         all_max = df.max(axis=0).max()
#         print(i, max_val, all_max)
