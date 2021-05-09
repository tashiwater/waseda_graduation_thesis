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
DATA_DIR = CURRENT_DIR + "/../../data/MTRNN/1127/noimg/"
INPUT_PATH = DATA_DIR + "result/"
output_fig_path = DATA_DIR + "result/"

with open(INPUT_PATH + "pca.pkl", mode="rb") as f:
    pca_base = pickle.load(f)
with open(INPUT_PATH + "pca_train.pickle", mode="rb") as f:
    pca = pickle.load(f)
components = pca_base.n_components

one_num = 159
container_num = 12
each_container = 3
stack_num = 6

stack = [
    pca[one_num * each_container * i : one_num * each_container * (i + 1)]
    for i in range(container_num)
]

mode = "online2"
theta = 30
cs_num = 12

if mode == "test":
    test_dir = DATA_DIR + "test_result2/"
    paths = [str(p) for p in Path(test_dir).glob("./*.xlsx")]
elif mode == "online":
    test_dir = (
        CURRENT_DIR + "/../../../../wiping_ws/src/wiping/online/data/1215log_ok/output/"
    )
    paths = [test_dir + "20201215_182803_cf80_cs8_type11_open03.csv"]
if mode == "test" or mode == "online":
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
    elif mode == "online":
        cs_start = 140
        test_np = datas[:, :, cs_start : cs_start + cs_num]
    test_np = test_np.reshape(-1, cs_num)
    test_pca = pca_base.transform(test_np)

    test_stack = [test_pca[one_num * i : one_num * (i + 1)] for i in range(stack_num)]


colorlist = ["r", "g", "b", "c", "m", "y", "k"]


# for i in range(185):
fig = plt.figure()
ax = Axes3D(fig)
for container in range(stack_num):
    for i in range(each_container):
        # start = ((container) * each_container + i) * one_num
        # datas = pca[start : start + one_num]
        # ax.plot(
        #     datas[:, 0],
        #     datas[:, 1],
        #     datas[:, 2],
        #     # label="{}".format(container),
        #     color=colorlist[container],
        # )
        start = ((container + stack_num) * each_container + i) * one_num
        datas = pca[start : start + one_num]
        ax.plot(
            datas[:, 0],
            datas[:, 1],
            datas[:, 2],
            # label="{}".format(container),
            color=colorlist[container],
        )
# start = 0
# datas = test_pca[start : start + one_num]
# ax.plot(
#     datas[:, 0],
#     datas[:, 1],
#     datas[:, 2],
#     # label="{}".format(container),
#     color=colorlist[-1],
# )
plt.xlabel("pca{} ({:.2})".format(0 + 1, pca_base.explained_variance_ratio_[0]))
plt.ylabel("pca{} ({:.2})".format(1 + 1, pca_base.explained_variance_ratio_[1]))
ax.set_zlabel("pca{} ({:.2})".format(2 + 1, pca_base.explained_variance_ratio_[2]))
plt.legend()
plt.show()
# fig.savefig(paths[0] + ".png")


# for i in range(components):
#     axis1 = i
#     start = 0
#     end = one_num * each_container
#     for j in range(components - i - 1):
#         axis2 = 1 + j + i
#         for k in range(stack_num):
#             plt.scatter(
#                 stack[k][start:end, axis1],
#                 stack[k][start:end, axis2],
#                 label="{}".format(k),
#                 edgecolors=colorlist[k],
#                 facecolor="None",
#                 marker="o",
#             )
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

#     test_start = 0
#     test_end = 119
#     if mode == "test":
#         plt.scatter(
#             test_stack[k][test_start:test_end, axis1],
#             test_stack[k][test_start:test_end, axis2],
#             label="test{}".format(k),
#             color=colorlist[k],
#             marker="D",
#         )
# if mode == "online":
#     plt.scatter(
#         test_pca[test_start:test_end, axis1],
#         test_pca[test_start:test_end, axis2],
#         # label="test{}".format(k),
#         color=colorlist[-1],
#         marker="D",
#     )
