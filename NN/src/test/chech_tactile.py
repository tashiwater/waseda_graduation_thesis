#!/usr/bin/env python3
# coding: utf-8

import os
from pathlib import Path
import pandas as pd
import numpy as np

import csv

# import matplotlib.pyplot as plt


if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = CURRENT_DIR + "/../../data/MTRNN/1210/size/"
    input_dir = DATA_DIR + "test/"
    result_path = DATA_DIR + "result/step_test.csv"
    paths = [str(p) for p in Path(input_dir).glob("./*.csv")]
    paths.sort()
    datas = []
    change_times = []
    for path in paths:
        df = pd.read_csv(path)
        df_tactile = df.iloc[:, 14:30]
        df_max = df_tactile.max(axis="columns")
        print(df_max)
        old_sign = -1
        change_time = []
        for i, val in enumerate(df_max):
            sign = np.sign(val)
            if sign != old_sign:
                change_time.append(i)
                old_sign = sign
        # print(df.shape)
        change_times.append(change_time)
    print(change_times)
    with open(result_path, "w", encoding="Shift_jis") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(change_times)
