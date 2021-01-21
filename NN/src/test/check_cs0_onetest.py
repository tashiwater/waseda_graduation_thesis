#!/usr/bin/env python3
# coding: utf-8
import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from pathlib import Path
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
path = CURRENT_DIR + "/../../data/cs0maker/0106/cs0.csv"
df = pd.read_csv(path, header=None, index_col=None)

container_num = 4
each_container = 3
components = 4
pca_base = PCA(n_components=components)
pca_train = pca_base.fit_transform(df)
pca_train = pca_train.reshape(container_num, each_container, components)

test_dir = (
    CURRENT_DIR
    + "/../../../../wiping_ws/src/wiping/online/data/0115log cs0maker/output/"
)
paths = [test_dir + "cs0_cf90_cs10_type04_open10_20210115_172142.csv"]
paths.sort()
datas = []
for path in paths:
    df = pd.read_csv(path)
    # print(df.shape)
    datas.append(df.values)
datas = np.array(datas)
# test_np = datas[:, :, 82:]
cs_num = pca_base.n_features_
cs_start = 60 + 90  # - 90
test_np = datas[:, 0, cs_start : cs_start + cs_num]
test_np = test_np.reshape(-1, cs_num)
pca_test = pca_base.transform(test_np)

# test_pca = test_pca.reshape(container_num, each_container, components)

label_list = [
    "Big Rectangular",
    "Middle Rectangular",
    "Middle Cylinder",
    "Small Cylinder",
]
colorlist = ["r", "g", "b", "c", "m", "y", "k"]
# for i in range(185):
fig = plt.figure()
for i in range(components):
    axis1 = i
    for j in range(components - i - 1):
        axis2 = 1 + j + i
        for k in range(container_num):
            plt.scatter(
                pca_train[k, :, axis1],
                pca_train[k, :, axis2],
                # label="{} 0deg".format(k),
                label=label_list[k],
                edgecolors=colorlist[k],
                facecolor="None",
                marker="o",
            )
        # plt.scatter(
        #     pca_test[:, axis1],
        #     pca_test[:, axis2],
        #     label="test",
        #     edgecolors=colorlist[-1],
        #     facecolor="None",
        #     marker="o",
        # )
        plt.xlabel(
            "pca{} ({:.2})".format(axis1 + 1, pca_base.explained_variance_ratio_[axis1])
        )
        plt.ylabel(
            "pca{} ({:.2})".format(axis2 + 1, pca_base.explained_variance_ratio_[axis2])
        )
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.title("cs0")
        plt.subplots_adjust(right=0.7)
        plt.show()
        # fig.savefig(paths[0] + ".png")
