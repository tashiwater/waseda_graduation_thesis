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


def load():
    paths = [str(p) for p in Path(train_dir).glob("./*.csv")]
    paths.sort()


img_start = 30
datas = []
components = 4
container_num = 4
each_container = 3

for path in paths:
    df = pd.read_csv(path)
    # df.values[:, cs_start : cs_start + cs_num]
    datas.append(df.values[0, img_start:])

# new_datas = []
# for df in datas:
#     new_datas.append(df[:step_min])
test_np = np.array(datas)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = CURRENT_DIR + "/../../data/MTRNN/0106/all/"
train_dir = DATA_DIR + "train"
paths = [str(p) for p in Path(train_dir).glob("./*.csv")]
paths.sort()
img_start = 30
datas = []
components = 4
container_num = 4
each_container = 3

for path in paths:
    df = pd.read_csv(path)
    # df.values[:, cs_start : cs_start + cs_num]
    datas.append(df.values[0, img_start:])

# new_datas = []
# for df in datas:
#     new_datas.append(df[:step_min])
test_np = np.array(datas)
# test_np = test_np.reshape(-1, cs_num)
step_min = 1
pca_base = PCA(n_components=components)
pca_cs = pca_base.fit_transform(test_np)
pca_cs = pca_cs.reshape(container_num, each_container, step_min, components)

colorlist = ["r", "g", "b", "c", "m", "y", "k"]
# colorlist = plt.get_cmap("tab10").colors
# for i in range(185):
fig = plt.figure()

show_3d = True
start = 0
end = 55
if show_3d:
    ax = Axes3D(fig)
    for container in range(container_num):
        n = container * each_container
        ax.scatter3D(
            pca_cs[container, :, start:end, 0],
            pca_cs[container, :, start:end, 1],
            pca_cs[container, :, start:end, 2],
            label="{}".format(container),
            color=colorlist[container],
            # s=1,
            # edgecolors=colorlist[container],
            # facecolor="None",
            # marker="o",
        )
    plt.xlabel("pca{} ({:.2})".format(0 + 1, pca_base.explained_variance_ratio_[0]))
    plt.ylabel("pca{} ({:.2})".format(1 + 1, pca_base.explained_variance_ratio_[1]))
    ax.set_zlabel("pca{} ({:.2})".format(2 + 1, pca_base.explained_variance_ratio_[2]))
    # plt.legend()
    plt.show()

for i in range(components):
    axis1 = i
    for j in range(components - i - 1):
        axis2 = 1 + j + i
        for container in range(container_num):
            k = container
            plt.scatter(
                pca_cs[container, :, start:end, axis1],
                pca_cs[container, :, start:end, axis2],
                label="{}".format(k),
                edgecolors=colorlist[k],
                facecolor="None",
                marker="o",
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
