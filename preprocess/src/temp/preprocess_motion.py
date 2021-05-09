#!/usr/bin/env python3
# coding: utf-8
import os
from pathlib import Path
import numpy as np


def mean_csv2csv(
    sequence_num, each_sample, class_num, foldername_func, raw_dir_name, result_dir_name
):
    for k in range(class_num):
        filename = foldername_func(k + 1)
        RAW_DIR = raw_dir_name + filename
        RESULT_DIR = result_dir_name + filename
        print(RAW_DIR)
        paths = [str(p) for p in Path(RAW_DIR).glob("./*.csv")]
        paths.sort()
        print(paths)

        for j, path in enumerate(paths):
            one_file = np.loadtxt(path, delimiter=",")
            sequence_num_raw = int(len(one_file) / each_sample)
            ret = [
                one_file[i * each_sample : i * each_sample + each_sample].mean(axis=0)
                for i in range(sequence_num_raw)
            ]
            # dummy_size = SEQUENCE_NUM - len(ret)
            for i in range(sequence_num - len(ret)):
                ret.append(ret[-1])
            ret = np.array(ret)
            np.savetxt(
                RESULT_DIR + "{:02}.csv".format(j + 1), ret, delimiter=",",
            )


if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = CURRENT_DIR + "/../data/"
    RAW_DIR = DATA_DIR + "tactile_raw/"
    RESULT_DIR = DATA_DIR + "tactile_preprocessed/"

    def foldername_func(i):
        return "tactile{}/".format(i + 1)

    mean_csv2csv(215, 4, 4, foldername_func, RAW_DIR, RESULT_DIR)


# SEQUENCE_NUM = 215
# EACH_SAMPLE = 4
# mode = "motion"
# CLASS_NUM = 4
# for class_num in range(CLASS_NUM):
#     if mode == "tactile":
#         filename = "tactile{}/".format(class_num + 1)
#         RAW_DIR = DATA_DIR + "tactile_raw/" + filename
#         RESULT_DIR = DATA_DIR + "tactile_preprocessed/" + filename
#     elif mode == "motion":
#         filename = "{:02}/".format(class_num + 1)
#         RAW_DIR = DATA_DIR + "motion_csv/" + filename
#         RESULT_DIR = DATA_DIR + "motion_preprocessed/" + filename
#     print(RAW_DIR)
#     paths = [str(p) for p in Path(RAW_DIR).glob("./*.csv")]
#     paths.sort()
#     print(paths)

#     for j, path in enumerate(paths):
#         one_file = np.loadtxt(path, delimiter=",")
#         sequence_num_raw = int(len(one_file) / EACH_SAMPLE)
#         ret = [
#             one_file[i * EACH_SAMPLE : i * EACH_SAMPLE + EACH_SAMPLE].mean(axis=0)
#             for i in range(sequence_num_raw)
#         ]
#         # dummy_size = SEQUENCE_NUM - len(ret)
#         for i in range(SEQUENCE_NUM - len(ret)):
#             ret.append(ret[-1])
#         ret = np.array(ret)
#         np.savetxt(
#             RESULT_DIR + "{:02}.csv".format(j + 1), ret, delimiter=",",
#         )
