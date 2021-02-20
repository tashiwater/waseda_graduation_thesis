#!/usr/bin/env python3
# coding: utf-8

import torch
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import os
from pathlib import Path


class DumpLossFig:
    def __init__(self):
        CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        DATA_DIR = CURRENT_DIR + "/../data/"
        paths = [str(p) for p in Path(DATA_DIR).glob("./*/*/*.csv")]
        paths += [str(p) for p in Path(DATA_DIR).glob("./*/*.csv")]
        print(paths)
        for path in paths:
            self.load_model(path)
            self.show_result()

    def load_model(self, csv_path):
        self._model_dir = csv_path
        df = pd.read_csv(csv_path)
        self._graph_epoch = list(df["epoch"])  # x dim of figure
        self._train_loss_value = list(df["train_loss"])
        self._test_loss_value = list(df["test_loss"])  # test loss list
        self._param_dict = {"epoch": self._graph_epoch[-1]}

    def show_result(self):
        """RESULT OUTPUT"""

        plt.figure(figsize=(6, 6))
        plt.plot(
            self._graph_epoch[10:], self._train_loss_value[10:], label="train loss"
        )
        if self._test_loss_value[10] != 0:
            plt.plot(
                self._graph_epoch[10:],
                self._test_loss_value[10:],
                label="test loss",
                c="#00ff00",
            )
        plt.xlim(1, self._param_dict["epoch"])
        # # plt.ylim(0, 0.2)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        # plt.title("loss")
        # plt.show()
        plt.savefig(self._model_dir + "loss_image.png")
        plt.clf()


if __name__ == "__main__":
    DumpLossFig()