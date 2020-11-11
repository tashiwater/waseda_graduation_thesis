#!/usr/bin/env python3
# coding: utf-8
import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import animation

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = CURRENT_DIR + "/../data/"
INPUT_PATH = DATA_DIR + "connected/"
# INPUT_PATH = "/home/user/TAKUMI_SHIMIZU/waseda_graduation_thesis/MTRNN/data/train/"
paths = [str(p) for p in Path(INPUT_PATH).glob("./*.csv")]
paths.sort()
datas = []
for path in paths:
    df = pd.read_csv(path)
    datas.append(df)
datas = np.array(datas)
imgs = datas[:, :, 26:]
# imgs = np.vstack(imgs)
imgs = imgs.reshape(-1, 15)
components = 5
pca = PCA(n_components=components).fit_transform(imgs)
circle_num = 185 * 12
circle = pca[:circle_num]
rectangle = pca[circle_num:]
# circle = pca[:4500]
# rectangle = pca[4500:]

one_num = 185
container_num = 6
stack_num = 6
stack = [0 for i in range(stack_num)]

# stack[0] = tuple([pca[one_num * i + 0 : one_num * i + 3] for i in range(container_num)])
# stack[1] = tuple([pca[one_num * i + 3 : one_num * i + 6] for i in range(container_num)])
# stack[2] = tuple([pca[one_num * i + 6 : one_num * i + 9] for i in range(container_num)])
# stack4 = tuple([pca[one_num * i + 0 : one_num * i + 3] for i in range(3)])


stack[0] = circle[:one_num]
stack[1] = circle[one_num * 4 : one_num * 5]
stack[2] = circle[one_num * 8 : one_num * 9]

stack[3] = rectangle[:one_num]
stack[4] = rectangle[one_num * 4 : one_num * 5]
stack[5] = rectangle[one_num * 8 : one_num * 9]

# stack[0] = circle
# stack[1] = rectangle

# data = [0 for i in range(stack_num)]
# for i in range(stack_num):
#     data[i] = np.vstack(stack[i])  # [3:]
# data2 = np.vstack(stack2)

fig, ax = plt.subplots()
colorlist = ["r", "g", "b", "c", "m", "y", "k", "w"]
# for i in range(185):
for i in range(components):
    axis1 = i
    for j in range(components - i - 1):
        axis2 = 1 + j + i
        for k in range(stack_num):
            ax.plot(
                stack[k][:30, axis1],
                stack[k][:30, axis2],
                label="{}".format(k),
                color=colorlist[k],
                marker="o",
            )

            ax.plot(
                stack[k][:1, axis1],
                stack[k][:1, axis2],
                # label="{}".format(k),
                color=colorlist[k],
                marker="x",
            )
        plt.xlabel("pca{}".format(axis1))
        plt.ylabel("pca{}".format(axis2))
        plt.legend()
        plt.show()
#     p = ax.plot(
#         stack[5][i, 0],
#         stack[5][i, 1],
#         label="{}".format(k),
#         color=colorlist[5],
#         marker="o",
#     )
#     ims.append(p)
# ani = animation.ArtistAnimation(fig, ims, interval=100)  # ArtistAnimationでアニメーションを作成する。


# ani.save("animate.gif", writer="imagemagick", dpi=300)  # gifで保存
# data5 = circle
# data6 = rectangle
# colorlist = ["r", "g", "b", "c", "m", "y", "k", "w"]
# for i in range(components):
#     axis1 = i
#     for j in range(components - i - 1):
#         axis2 = 1 + j + i
#         for k in range(stack_num):
#             plt.scatter(
#                 data[k][:, axis1],
#                 data[k][:, axis2],
#                 label="{}".format(k),
#                 color=colorlist[k],
#                 marker="o",
#             )

# plt.scatter(
#     data1[:, axis1], data1[:, axis2], label="1", color="red", marker="o"
# )
# plt.scatter(
#     data2[:, axis1], data2[:, axis2], label="2", color="blue", marker="o"
# )
# plt.scatter(
#     data3[:, axis1], data3[:, axis2], label="3", color="green", marker="o"
# )

# plt.scatter(data4[:, axis1], data4[:, axis2], color="purple", marker="o")
# plt.scatter(data5[:, axis1], data5[:, axis2], color="black", marker="o")
# plt.scatter(data6[:, axis1], data6[:, axis2], color="yellow", marker="o")

# plt.xlabel("pca{}".format(axis1))
# plt.ylabel("pca{}".format(axis2))

# plt.show()
