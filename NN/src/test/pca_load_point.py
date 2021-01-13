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

step_df = pd.read_csv(DATA_DIR + "step_test_end.csv")
test_step_np = pd.read_csv(DATA_DIR + "step_test2_end.csv").values
step_np = step_df.values
step_title = step_df.columns
# print(step_title)
with open(INPUT_PATH + "pca.pkl", mode="rb") as f:
    pca_base = pickle.load(f)
with open(INPUT_PATH + "pca_train.pickle", mode="rb") as f:
    pca_train = pickle.load(f)

components = pca_base.n_components

one_num = 139
container_num = 4
each_container = 3
stack_num = container_num

pca_train = pca_train.reshape(-1, one_num, components)

mode = "online2"
if mode == "test":
    test_dir = DATA_DIR + "result/"
    paths = [str(p) for p in Path(test_dir).glob("./*.xlsx")]
elif mode == "online":
    test_dir = (
        CURRENT_DIR + "/../../../../wiping_ws/src/wiping/online/data/0107log/output/"
    )
    paths = [test_dir + "cf90_cs10_type03_open08_20210107_145151.csv"]
elif mode == "online2":
    test_len = 139
    test_dir = (
        CURRENT_DIR
        + "/../../../../wiping_ws/src/wiping/online/data/0109log/output/online/"
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
        temp = df.values[:test_len]
        pad_width = ((0, test_len - len(temp)), (0, 0))
        temp = np.pad(temp, pad_width, "constant", constant_values=0)
        datas.append(temp[:test_len])
    datas = np.array(datas)
    # test_np = datas[:, :, 82:]
    cs_num = pca_base.n_features_
    cs_start = 60  # $+ 90  # - 90
    test_np = datas[:, :, cs_start : cs_start + cs_num]
    test_np = test_np.reshape(-1, cs_num)
    test_pca = pca_base.transform(test_np)
    test_pca = test_pca.reshape(len(paths), -1, components)

point_list = []
for i, container in enumerate(pca_train):
    point_list.append(container[step_np[i]])
point_np = np.array(point_list)


test_point_list = []
for i, container in enumerate(test_pca):
    test_point_list.append(container[test_step_np[i]])
test_point_np = np.array(test_point_list)

colorlist = plt.get_cmap("tab10").colors
# colorlist = ["r", "g", "b", "c", "m", "y", "k"]
fig = plt.figure()

show_3d = True
start = 0
end = one_num
if show_3d:
    ax = Axes3D(fig)
    for container in range(stack_num):
        n = container * each_container
        ax.scatter3D(
            pca_train[n : n + each_container, start:end, 0],
            pca_train[n : n + each_container, start:end, 1],
            pca_train[n : n + each_container, start:end, 2],
            label="container_{}(offline)".format(container),
            color=colorlist[container],
            s=1,
            # edgecolors=colorlist[container],
            # facecolor="None",
            # marker="o",
        )
    for point in range(point_np.shape[1]):
        ax.scatter3D(
            point_np[start:end, point, 0],
            point_np[start:end, point, 1],
            point_np[start:end, point, 2],
            label="surface" + step_title[point] + "(offline)",
            color=colorlist[point],
            # s=1,
            edgecolors=colorlist[-1],
            # facecolor="None",
            # marker="o",
        )
        # if mode == "online2":
        #     ax.scatter3D(
        #         test_point_np[:, point, 0],
        #         test_point_np[:, point, 1],
        #         test_point_np[:, point, 2],
        #         label="surface" + step_title[point] + "(online)",
        #         color=colorlist[point],
        #         edgecolors="black",
        #         marker="D",
        #     )
        # if mode == "online2":
        #     test_np = datas[n][:, cs_start : cs_start + cs_num]
        #     test_np = test_np.reshape(-1, cs_num)
        #     test_pca = pca_base.transform(test_np)
        #     ax.scatter(
        #         test_pca[start:end, 0],
        #         test_pca[start:end, 1],
        #         test_pca[start:end, 2],
        #         label="online{}".format(container),
        #         color=colorlist[container],
        #         # edgecolors=colorlist[-1],
        #         # s=3,
        #         marker="D",
        #     )

    plt.xlabel("pca{} ({:.2})".format(0 + 1, pca_base.explained_variance_ratio_[0]))
    plt.ylabel("pca{} ({:.2})".format(1 + 1, pca_base.explained_variance_ratio_[1]))
    ax.set_zlabel("pca{} ({:.2})".format(2 + 1, pca_base.explained_variance_ratio_[2]))
    plt.legend()
    plt.show()

for i in range(components):
    axis1 = i
    # if axis1 != 0:
    #     continue
    for j in range(components - i - 1):
        axis2 = 1 + j + i
        # if axis2 != 2:
        # continue
        for point in range(point_np.shape[1]):
            plt.scatter(
                point_np[start:end, point, axis1],
                point_np[start:end, point, axis2],
                label="surface" + step_title[point] + "(offline)",
                color=colorlist[point + 5],
                marker="o",
            )
            if mode == "online2":
                plt.scatter(
                    test_point_np[:, point, axis1],
                    test_point_np[:, point, axis2],
                    label="surface" + step_title[point] + "(online)",
                    color=colorlist[point + 5],
                    edgecolors="black",
                    marker="D",
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
            # if mode == "test":
            #     test_start = 0
            #     test_end = 159
            #     plt.scatter(
            #         test_stack[k][test_start:test_end, axis1],
            #         test_stack[k][test_start:test_end, axis2],
            #         # label="test{}".format(k),
            #         edgecolors=colorlist[k],
            #         facecolor="None",
            #         # color=colorlist[k],
            #         marker="D",
            #     )
            # elif mode == "online2":
            #     n = k + 6
            #     test_np = datas[n][:, cs_start : cs_start + cs_num]
            #     test_np = test_np.reshape(-1, cs_num)
            #     test_pca = pca_base.transform(test_np)
            #     plt.scatter(
            #         test_pca[start:end, axis1],
            #         test_pca[start:end, axis2],
            #         label="online{}".format(k),
            #         edgecolors=colorlist[-1],
            #         facecolor=colorlist[k],
            #         marker="D",
            #     )
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
