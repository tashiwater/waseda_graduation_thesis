#!/usr/bin/env python3
# coding: utf-8
import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from pathlib import Path
import matplotlib.pyplot as plt
import pickle

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = CURRENT_DIR + "/../../data/MTRNN/1210/size/"
INPUT_PATH = DATA_DIR + "train_result/"
output_fig_path = DATA_DIR + "result/"

with open(INPUT_PATH + "pca.pkl", mode="rb") as f:
    pca_base = pickle.load(f)
with open(INPUT_PATH + "pca_train.pickle", mode="rb") as f:
    pca = pickle.load(f)
components = 2

one_num = 119
container_num = 4
each_container = 3
circle_num = one_num * container_num * each_container // 2
circle = pca[:circle_num]
rectangle = pca[circle_num:]
# circle = pca[:4500]
# rectangle = pca[4500:]

stack_num = container_num
# stack[0] = tuple([pca[one_num * i + 0 : one_num * i + 3] for i in range(container_num)])
# stack[1] = tuple([pca[one_num * i + 3 : one_num * i + 6] for i in range(container_num)])
# stack[2] = tuple([pca[one_num * i + 6 : one_num * i + 9] for i in range(container_num)])
# stack4 = tuple([pca[one_num * i + 0 : one_num * i + 3] for i in range(3)])

stack = [
    pca[one_num * each_container * i : one_num * each_container * (i + 1)]
    for i in range(container_num)
]

mode = "online"
cs_num = 90

if mode == "test":
    test_dir = DATA_DIR + "test_result2/"
    paths = [str(p) for p in Path(test_dir).glob("./*.xlsx")]
elif mode == "online":
    test_dir = CURRENT_DIR + "/../../../../wiping_ws/src/wiping/online/data/log/output/"
    paths = [test_dir + "20201213_144140_cf90_cs10_type0_open01.csv"]

paths.sort()
datas = []
for path in paths:
    if mode == "test":
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    # print(df.shape)
    datas.append(df.values)
datas = np.array(datas)
# test_np = datas[:, :, 82:]
if mode == "test":
    cs_start = 68
    test_np = datas[:, :, cs_start : cs_num + cs_start]
else:
    cs_start = 62
    test_np = datas[:, :, cs_start:cs_start + cs_num]
test_np = test_np.reshape(-1, cs_num)
test_pca = pca_base.transform(test_np)

test_stack = [test_pca[one_num * i : one_num * (i + 1)] for i in range(stack_num)]


colorlist = ["r", "g", "b", "c", "m", "y", "k"]


# for i in range(185):
fig = plt.figure()
for i in range(components):
    axis1 = i
    start = 0
    end = one_num * each_container
    for j in range(components - i - 1):
        axis2 = 1 + j + i
        for k in range(stack_num):
            plt.scatter(
                stack[k][start:end, axis1],
                stack[k][start:end, axis2],
                label="{}_theta0".format(k),
                edgecolors=colorlist[k],
                facecolor="None",
                marker="o",
            )
            # plt.scatter(
            #     stack[k + stack_num // 2][start:end, axis1],
            #     stack[k + stack_num // 2][start:end, axis2],
            #     label="{}_theta30".format(k),
            #     edgecolors=colorlist[k],
            #     facecolor="None",
            #     marker="D",
            # )

            # plt.plot(
            #     stack[k][start : start + 1, axis1],
            #     stack[k][start : start + 1, axis2],
            #     # label="{}".format(k),
            #     color=colorlist[k],
            #     marker="D",
            # )

            test_start = 0
            test_end = 119
            if mode == "test":
                plt.scatter(
                    test_stack[k][test_start:test_end, axis1],
                    test_stack[k][test_start:test_end, axis2],
                    label="test{}".format(k),
                    color=colorlist[k],
                    marker="D",
                )
        if mode == "online":
            plt.scatter(
                test_pca[test_start:test_end, axis1],
                test_pca[test_start:test_end, axis2],
                # label="test{}".format(k),
                color=colorlist[-1],
                marker="D",
            )
        plt.xlabel("pca{}".format(axis1 + 1))
        plt.ylabel("pca{}".format(axis2 + 1))
        plt.legend()
        plt.show()
        fig.savefig(paths[0] + ".png")
