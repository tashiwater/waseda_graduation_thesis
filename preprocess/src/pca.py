#!/usr/bin/env python3
# coding: utf-8
import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from pathlib import Path
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = CURRENT_DIR + "/../data/"
INPUT_PATH = DATA_DIR + "connected/"

paths = [str(p) for p in Path(INPUT_PATH).glob("./*.csv")]
paths.sort()
datas = []
for path in paths:
    df = pd.read_csv(path)
    datas.append(df)
datas = np.array(datas)
imgs = datas[:, 0:1, 30:]
imgs = np.vstack(imgs)
components = 6
pca = PCA(n_components=components).fit_transform(imgs)
circle_num = 27
circle = pca[:circle_num]
rectangle = pca[circle_num:]
# circle = pca[:4500]
# rectangle = pca[4500:]

one_num = 9
container_num = 9
"""
stack1 = tuple([pca[one_num * i + 0 : one_num * i + 3] for i in range(container_num)])
stack2 = tuple([pca[one_num * i + 3 : one_num * i + 6] for i in range(container_num)])
stack3 = tuple([pca[one_num * i + 6 : one_num * i + 9] for i in range(container_num)])
# stack4 = tuple([pca[one_num * i + 0 : one_num * i + 3] for i in range(3)])
"""
stack1 = circle[:9]
stack2 = circle[9:18]
stack3 = circle  # [18:]

stack4 = rectangle[:9]
stack5 = rectangle[9:18]
stack6 = rectangle  # [18:]


data1 = np.vstack(stack1)[3:]
data2 = np.vstack(stack2)[3:]
data3 = np.vstack(stack3)[3:]
data4 = np.vstack(stack4)
data5 = np.vstack(stack5)
data6 = np.vstack(stack6)

# data1 = circle
# data2 = rectangle

for i in range(components):
    axis1 = i
    for j in range(components - i - 1):
        axis2 = 1 + j + i
        print(axis1, axis2)
        plt.scatter(
            data1[:, axis1], data1[:, axis2], label="30", color="red", marker="o"
        )
        plt.scatter(
            data2[:, axis1], data2[:, axis2], label="0", color="blue", marker="o"
        )
        plt.scatter(
            data3[:, axis1], data3[:, axis2], label="-30", color="green", marker="o"
        )

        plt.scatter(data4[:, axis1], data4[:, axis2], color="purple", marker="o")
        plt.scatter(data5[:, axis1], data5[:, axis2], color="black", marker="o")
        plt.scatter(data6[:, axis1], data6[:, axis2], color="yellow", marker="o")

        plt.xlabel("pca{}".format(axis1))
        plt.ylabel("pca{}".format(axis2))
        # plt.legend()
        plt.show()
print(data1)
