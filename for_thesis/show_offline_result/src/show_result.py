#!/usr/bin/env python3
# coding:utf-8

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path


class ShowResult:
    def get_labels(self, add_word, n=0):
        return (
            [add_word + "position{}".format(i + n) for i in range(7)]
            + [add_word + "torque{}".format(i + n) for i in range(7)]
            + [add_word + "tactile{}".format(i) for i in range(16)]
            # + [add_word + "image{}".format(i) for i in range(15)]
        )

    def get_labels2(self, add_word, n=0):
        return (
            [add_word + "joint{}".format(i + n) for i in range(7)]
            + [add_word + "torque{}".format(i + n) for i in range(7)]
            + [add_word + "tactile{}".format(i) for i in range(16)]
            # + [add_word + "image{}".format(i) for i in range(15)]
        )

    def rename(self, df):
        name_dict = dict(
            zip(
                self.get_labels("in ") + self.get_labels("out "),
                self.get_labels2("teach ", 1) + self.get_labels2("predict ", 1),
            )
        )
        return df.rename(columns=name_dict)

    def run(self, data_dir, layer_list):

        in_size = sum(layer_list)
        paths = [str(p) for p in Path(data_dir).glob("./*.csv")]
        for path in paths:
            last = 0
            for i, layer in enumerate(layer_list):
                df = pd.read_csv(path)
                df = self.rename(df)
                # if i >= 2:
                #     fig, ax = plt.subplots(figsize=(10, 8))
                # else:
                # fig, ax = plt.subplots(figsize=(8, 4))
                fig, ax = plt.subplots(figsize=(8, 3))
                colormap = "tab20"
                legend = None
                df.iloc[:, last : last + layer].plot(
                    colormap=colormap, linestyle="--", ax=ax, legend=legend
                )
                df.iloc[:, in_size + last : in_size + last + layer].plot(
                    colormap=colormap, ax=ax, legend=legend
                )
                last += layer
                plt.ylim(-1, 1)
                plt.xlabel("step")
                plt.ylabel("normalized position")
                # plt.legend(loc="upper left", bbox_to_anchor=(1, 1))

                # plt.subplots_adjust(right=0.7)
                # plt.subplots_adjust(left=0.1, right=0.7, bottom=0.1, top=0.95)
                # plt.show()
                fig.savefig(path + "{}.png".format(i))
                # plt.cl()


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = current_dir + "/../data/all/"
    layer_list = [7, 7, 16, 15]
    show_result = ShowResult()

    data_dir = current_dir + "/../data/normal/"
    layer_list = [7, 7, 16]
    show_result.run(data_dir, layer_list)
    # data_dir = current_dir + "/../data/all/"
    # layer_list = [7, 7, 16, 15]
    # show_result.run(data_dir, layer_list)
    # data_dir = current_dir + "/../data/cs0/"
    # layer_list = [7, 7, 16]
    # show_result.run(data_dir, layer_list)
