#!/usr/bin/env python3
# coding: utf-8
import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = CURRENT_DIR + "/../../data/MTRNN/0106/normal/"
INPUT_PATH = DATA_DIR + "result/"
output_fig_path = DATA_DIR + "result/"

with open(INPUT_PATH + "pca.pkl", mode="rb") as f:
    pca_base = pickle.load(f)
with open(INPUT_PATH + "pca_train.pickle", mode="rb") as f:
    pca = pickle.load(f)

components = pca_base.n_components

one_num = 139
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
if mode == "test":
    test_dir = DATA_DIR + "result/"
    paths = [str(p) for p in Path(test_dir).glob("./*.xlsx")]
elif mode == "online":
    test_dir = (
        CURRENT_DIR + "/../../../../wiping_ws/src/wiping/online/data/0107log/output/"
    )
    paths = [test_dir + "cf90_cs10_type03_open08_20210107_145151.csv"]
elif mode == "online2":
    test_dir = (
        CURRENT_DIR
        + "/../../../../wiping_ws/src/wiping/online/data/1223log9006/output/"
    )
    paths = [str(p) for p in Path(test_dir).glob("./*.csv")]

if mode == "test" or mode == "online" or mode == "online2":
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
    cs_num = pca_base.n_features_
    cs_start = 60 + 90  # - 90
    if mode != "online2":
        test_np = datas[:, :, cs_start : cs_start + cs_num]
        test_np = test_np.reshape(-1, cs_num)
        test_pca = pca_base.transform(test_np)

        test_stack = [
            test_pca[one_num * i : one_num * (i + 1)] for i in range(stack_num)
        ]


colorlist = ["r", "g", "b", "c", "m", "y", "k"]
# for i in range(185):
fig = plt.figure()

show_3d = True
start = 0
end = one_num * each_container
if show_3d:
    ax = Axes3D(fig)
    for container in range(stack_num):
        n = container
        ax.scatter3D(
            stack[n][start:end, 0],
            stack[n][start:end, 1],
            stack[n][start:end, 2],
            label="{}_theta0".format(container),
            color=colorlist[container],
            s=1,
            # edgecolors=colorlist[container],
            # facecolor="None",
            # marker="o",
        )
        if mode == "online2":
            test_np = datas[n][:, cs_start : cs_start + cs_num]
            test_np = test_np.reshape(-1, cs_num)
            test_pca = pca_base.transform(test_np)
            ax.scatter(
                test_pca[start:end, 0],
                test_pca[start:end, 1],
                test_pca[start:end, 2],
                label="online{}".format(container),
                color=colorlist[container],
                # edgecolors=colorlist[-1],
                # s=3,
                marker="D",
            )
        # for i in range(each_container):
        # start = ((container) * each_container + i) * one_num
        # datas = pca[start : start + one_num]
        # ax.plot(
        #     datas[:, 0],
        #     datas[:, 1],
        #     datas[:, 2],
        #     # label="{}".format(container),
        #     color=colorlist[container],
        # )
        # start = ((container + stack_num) * each_container + i) * one_num
        # datas = pca[start : start + one_num]
        # ax.plot(
        #     datas[:, 0],
        #     datas[:, 1],
        #     datas[:, 2],
        #     # label="{}".format(container),
        #     color=colorlist[container],
        # )
    if mode == "online":
        datas = test_pca[start : start + one_num]
        ax.plot(
            datas[start:end, 0],
            datas[start:end, 1],
            datas[start:end, 2],
            # label="{}".format(container),
            color=colorlist[-1],
        )
    plt.xlabel("pca{} ({:.2})".format(0 + 1, pca_base.explained_variance_ratio_[0]))
    plt.ylabel("pca{} ({:.2})".format(1 + 1, pca_base.explained_variance_ratio_[1]))
    ax.set_zlabel("pca{} ({:.2})".format(2 + 1, pca_base.explained_variance_ratio_[2]))
    # plt.legend()
    plt.show()

for i in range(components):
    axis1 = i
    # if axis1 != 0:
    #     continue
    for j in range(components - i - 1):
        axis2 = 1 + j + i
        # if axis2 != 2:
        # continue
        for k in range(stack_num):
            n = k
            plt.scatter(
                stack[n][start:end, axis1],
                stack[n][start:end, axis2],
                label="{}_theta0".format(k),
                edgecolors=colorlist[k],
                facecolor="None",
                marker="o",
            )

            # plt.scatter(
            #     stack[k + stack_num][start:end, axis1],
            #     stack[k + stack_num][start:end, axis2],
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
            if mode == "test":
                test_start = 0
                test_end = 159
                plt.scatter(
                    test_stack[k][test_start:test_end, axis1],
                    test_stack[k][test_start:test_end, axis2],
                    # label="test{}".format(k),
                    edgecolors=colorlist[k],
                    facecolor="None",
                    # color=colorlist[k],
                    marker="D",
                )
            elif mode == "online2":
                n = k + 6
                test_np = datas[n][:, cs_start : cs_start + cs_num]
                test_np = test_np.reshape(-1, cs_num)
                test_pca = pca_base.transform(test_np)
                plt.scatter(
                    test_pca[start:end, axis1],
                    test_pca[start:end, axis2],
                    label="online{}".format(k),
                    edgecolors=colorlist[-1],
                    facecolor=colorlist[k],
                    marker="D",
                )
        if mode == "online":
            test_start = start
            test_end = end
            plt.scatter(
                test_pca[test_start:test_end, axis1],
                test_pca[test_start:test_end, axis2],
                # label="test{}".format(k),
                color=colorlist[-1],
                marker="D",
            )
        plt.xlabel(
            "pca{} ({:.2})".format(axis1 + 1, pca_base.explained_variance_ratio_[axis1])
        )
        plt.ylabel(
            "pca{} ({:.2})".format(axis2 + 1, pca_base.explained_variance_ratio_[axis2])
        )
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.subplots_adjust(right=0.7)
        plt.show()
        # fig.savefig(paths[0] + ".png")
