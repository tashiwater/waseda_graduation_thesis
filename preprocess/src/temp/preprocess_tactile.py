#!/usr/bin/env python3
# coding: utf-8
import os
from pathlib import Path
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = CURRENT_DIR + "/../data/"
SEQUENCE_NUM = 215

for class_num in range(4):
    filename = "tactile{}/".format(class_num + 1)
    RAW_DIR = DATA_DIR + "tactile_raw/" + filename
    RESULT_DIR = DATA_DIR + "tactile_preprocessed/" + filename
    print(RAW_DIR)
    paths = [str(p) for p in Path(RAW_DIR).glob("./*.csv")]
    paths.sort()
    print(paths)

    for j, path in enumerate(paths):
        one_file = np.loadtxt(path, delimiter=",")
        sequence_num_raw = int(len(one_file) / 4)
        ret = [
            one_file[i * 4 : i * 4 + 4].mean(axis=0) for i in range(sequence_num_raw)
        ]
        # dummy_size = SEQUENCE_NUM - len(ret)
        for i in range(SEQUENCE_NUM - len(ret)):
            ret.append(ret[-1])
        ret = np.array(ret)
        np.savetxt(
            RESULT_DIR + "{:02}.csv".format(j + 1), ret, delimiter=",",
        )
