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


class PcaLoadBase:
    def __init__(self):
        self._components = 4
        self._pca_base = PCA(n_components=self._components)

    def set_params(self, container_num, step_min, read_start, read_num):
        self._container_num = container_num
        self._step_min = step_min
        self._read_start = read_start
        self._read_num = read_num

    def load(self, path, each_container, is_train):
        paths = [str(p) for p in Path(path).glob("./*.csv")]
        paths.sort()

        datas = []
        step_min = self._step_min
        read_start = self._read_start
        read_num = self._read_num

        for path in paths:
            df = pd.read_csv(path)
            df_np = df.values
            datas.append(df_np[:step_min, read_start : read_start + read_num])

        test_np = np.array(datas)
        test_np = test_np.reshape(-1, read_num)
        if is_train:
            pca_cs = self._pca_base.fit_transform(test_np)
            pca_cs = pca_cs.reshape(
                self._container_num, each_container, step_min, self._components
            )
            self._train_pca = pca_cs
        else:
            pca_cs = self._pca_base.transform(test_np)
            pca_cs = pca_cs.reshape(
                self._container_num, each_container, step_min, self._components
            )
            self._test_pca = pca_cs

    def draw3d(self, data, container):
        colorlist = ["r", "g", "b", "c", "m", "y", "k"]
        temp = data[container]
        self._ax.scatter3D(
            temp[0],
            temp[1],
            temp[2],
            label="{}".format(container),
            color=colorlist[container],
            # s=1,
            # edgecolors=colorlist[container],
            # facecolor="None",
            # marker="o",
        )

    def drawlabel(self, is_3d):
        ratio = self._pca_base.explained_variance_ratio_
        plt.xlabel("pca{} ({:.2})".format(0 + 1, ratio[0]))
        plt.ylabel("pca{} ({:.2})".format(1 + 1, ratio[1]))
        if is_3d:
            self._ax.set_zlabel("pca{} ({:.2})".format(2 + 1, ratio[2]))

    def run(self):
        colorlist = ["r", "g", "b", "c", "m", "y", "k"]
        # colorlist = plt.get_cmap("tab10").colors
        fig = plt.figure()
        show_3d = False
        start = 0
        end = 55
        if show_3d:
            self._ax = Axes3D(fig)
            for container in range(self._container_num):
                self.draw3d(self._train_pca)
                self.draw3d(self._test_pca)
            self.drawlabel(True)
            # plt.legend()
            plt.show()

        for i in range(self._components):
            axis1 = i
            for j in range(self._components - i - 1):
                axis2 = 1 + j + i
                for container in range(self._container_num):
                    k = container
                    plt.scatter(
                        self._train_pca[container, :, start:end, axis1],
                        self._train_pca[container, :, start:end, axis2],
                        label="{}".format(k),
                        edgecolors=colorlist[k],
                        facecolor="None",
                        marker="o",
                    )
                    plt.scatter(
                        self._test_pca[container, :, start:end, axis1],
                        self._test_pca[container, :, start:end, axis2],
                        # label="{}".format(k),
                        edgecolors=colorlist[k],
                        facecolor="None",
                        marker="o",
                    )
                self.drawlabel(False)
                plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
                plt.subplots_adjust(right=0.7)
                plt.show()
                # fig.savefig(paths[0] + ".png")
