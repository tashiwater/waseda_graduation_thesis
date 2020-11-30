#!/usr/bin/env python3
# coding: utf-8
import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from pathlib import Path
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
path = CURRENT_DIR + "/../../data/MTRNN_cs/result/out.csv"
df = pd.read_csv(path, header=None, index_col=None)

components = 4
pca_base = PCA(n_components=components)
pca_cs = pca_base.fit_transform(df)

container_num = 6
each_container = 3
theta_num = 2

stack = [pca_cs[i * each_container : (i + 1) * each_container] for i in range(6 * 2)]

colorlist = ["r", "g", "b", "c", "m", "y", "k"]
# for i in range(185):
fig = plt.figure()
for i in range(components):
    axis1 = i
    for j in range(components - i - 1):
        axis2 = 1 + j + i
        for k in range(container_num):
            plt.scatter(
                stack[2 * k][:, axis1],
                stack[2 * k][:, axis2],
                label="{} theta0".format(k),
                edgecolors=colorlist[k],
                facecolor="None",
                marker="o",
            )

            n = 2 * k + 1
            plt.scatter(
                stack[n][:, axis1],
                stack[n][:, axis2],
                label="{} theta30".format(k),
                edgecolors=colorlist[k],
                facecolor="None",
                marker="D",
            )

        plt.xlabel("pca{}".format(axis1 + 1))
        plt.ylabel("pca{}".format(axis2 + 1))
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.title("cs0")
        plt.subplots_adjust(right=0.7)
        plt.show()
        # fig.savefig(paths[0] + ".png")
