#!/usr/bin/env python3
# coding: utf-8
import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


class Compare:
    def read_data(self, path, prefix):
        extention = os.path.splitext(path)[-1]
        if extention == ".csv":
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path)
        return df.add_prefix(prefix)

    def read_offline(self):
        CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        DATA_DIR = CURRENT_DIR + "/../../data/MTRNN/1127/noimg/"
        self._result_dir = DATA_DIR + "result/"
        online_path = DATA_DIR + "result/output{:02d}.xlsx".format(
            self._container * 3 + 1
        )
        self._offline_df = self.read_data(online_path, "train_")

    def read_online(self, name):
        CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        offline_dir = (
            CURRENT_DIR
            + "/../../../../wiping_ws/src/wiping/online/data/1215log_ok/output/"
        )
        path = offline_dir + name + ".csv"
        self._online_df = self.read_data(path, "online_")

    def show(self, start, length):
        end = start + length

        step_end = self._online_df.shape[0] - 5
        colormap = "tab20"
        ax = self._offline_df.iloc[:step_end, start:end].plot(
            colormap=colormap, linestyle="--"
        )
        self._online_df.iloc[:step_end, start:end].plot(colormap=colormap, ax=ax)
        # plt.plot(df_output.iloc[:, in_size : in_size + 7], linestyle="-")
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.subplots_adjust(right=0.7)
        plt.xlabel("step")
        png_path = self._result_dir + "{}start{}.jpg".format(self._container, start)
        plt.savefig(png_path)
        plt.show()

    def init(self, container, name):
        self._container = container
        self.read_offline()
        self.read_online(name)


if __name__ == "__main__":
    main = Compare()
    main.init(1, "20201215_181038_cf80_cs8_type1_open03")
    main.show(0, 7)
    main.show(14, 16)
