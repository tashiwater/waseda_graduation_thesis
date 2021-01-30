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
test_dir = (
    CURRENT_DIR + "/../../../../wiping_ws/src/wiping/online/data/all_train/output/"
)
paths = [str(p) for p in Path(test_dir).glob("./*.csv")]
paths.sort()
datas = []
step_min = 200
cs_num = 10
cf_num = 90
io_num = 50
in_num = 45
cs_start = in_num * 2 + cf_num
components = 4
container_num = 4
each_container = 5

for path in paths:
    df = pd.read_csv(path)
    step = df.values.shape[0]
    # if step < step_min:
    #     step_min = step
    remain = step_min - step
    df_np = df.values
    for _ in range(remain):
        df_np = np.vstack([df_np, df_np[-1]])
    # df.values[:, cs_start : cs_start + cs_num]
    datas.append(df_np[:, cs_start : cs_start + cs_num])

# new_datas = []
# for df in datas:
#     new_datas.append(df[:step_min])
test_np = np.array(datas)
test_np = test_np.reshape(-1, cs_num)

pca_base = PCA(n_components=components)
pca_cs = pca_base.fit_transform(test_np)
pca_cs = pca_cs.reshape(container_num, each_container, step_min, components)

colorlist = ["r", "g", "b", "c", "m", "y", "k"]
# colorlist = plt.get_cmap("tab10").colors
# for i in range(185):
fig = plt.figure()

show_3d = True
start = 0
end = -1
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
        for container in range(stack_num):
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
