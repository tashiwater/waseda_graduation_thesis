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
    DATA_DIR = CURRENT_DIR + "/../../data/MTRNN/0106/normal/"
    # DATA_DIR = (
    #     CURRENT_DIR
    #     + "/../../../../wiping_ws/src/wiping/online/data/1223log9006/output/"
    # )
    input_dir = DATA_DIR + "train/"
    test_mode = True
    if test_mode:
        result_path = DATA_DIR + "step_test2.csv"
        test_dir = (
            CURRENT_DIR
            + "/../../../../wiping_ws/src/wiping/online/data/0109log/output/online/"
        )
        # paths = [test_dir + "onlinecf90_cs10_type00_open10_20210109_112454.csv"]
        paths = [str(p) for p in Path(test_dir).glob("./*.csv")]
    else:
        result_path = DATA_DIR + "step_test.csv"
        paths = [str(p) for p in Path(input_dir).glob("./*.csv")]
    paths.sort()
    print(paths)
    datas = []
    change_times = []
    for path in paths:
        df = pd.read_csv(path)
        df_tactile = df.iloc[:, 14:30]
        df_max = df_tactile.max(axis="columns")
        # print(df_max)
        old_sign = 0
        change_time = []
        for i, val in enumerate(df_max):
            # sign = np.sign(val)
            sign = 1 if val > -0.5 else 0
            if sign != old_sign:
                change_time.append(i)
                old_sign = sign
        # print(df.shape)
        change_times.append(change_time)
    print(change_times)
    titles = []
    for i in range(5):
        titles += ["{}start".format(i), "{}end".format(i)]
    with open(result_path, "w", encoding="Shift_jis") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(titles)  # add
        writer.writerows(change_times)
