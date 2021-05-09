#!/usr/bin/env python3
# coding: utf-8
import torch
from pathlib import Path
import numpy as np
import pandas as pd


class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, dir_path):
        super(MyDataSet, self).__init__()
        self._paths = [str(p) for p in Path(dir_path).glob("./*.csv")]
        self._paths.sort()

        self._datas = []
        for path in self._paths:
            df = pd.read_csv(path)
            x = torch.tensor(df.values.astype(np.float32))
            self._datas.append(x)
        self._len = len(self._datas)

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        # print(self._datas[index][1:].shape)
        return self._datas[index][0:-1], self._datas[index][1:]
