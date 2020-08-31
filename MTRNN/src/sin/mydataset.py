#!/usr/bin/env python3
# coding: utf-8
import torch
import torchvision
from pathlib import Path
import math
import numpy as np
import pandas as pd


class CsvDataSet(torch.utils.data.Dataset):
    def __init__(self, dir_path):
        super().__init__()
        # self._noise = noise
        self._paths = [str(p) for p in Path(dir_path).glob("./*/*.csv")]
        self._paths.sort()
        # self._datas = [
        #     torch.from_numpy(np.loadtxt(p, delimiter=",")).float() for p in self._paths
        # ]
        self._datas = []
        for path in self._paths:
            # print(path)
            df = pd.read_csv(path)
            # print(df)
            # x = torch.from_numpy(np.loadtxt(path, delimiter=",")).float()
            x = torch.tensor(df.values.astype(np.float32))
            self._datas.append(x)

        self._len = len(self._datas)

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        # print(self._datas[index][1:].shape)
        return self._datas[index][0:-1], self._datas[index][1:]


class EasyDataSet(torch.utils.data.Dataset):
    def __init__(self, start, data_num, step, batch_num, noise=0):
        x_datas, y_datas = [], []
        # funcs = [math.sin, math.cos, self.target_func]
        for i in range(batch_num):
            x, y = self.make_data(start, data_num, step, noise, self.target_func)
            x_datas.append(x)
            y_datas.append(y)

        self.ret, self.ret2 = x_datas, y_datas
        # torch.tensor(x_datas), torch.tensor(y_datas)
        self._len = len(self.ret)

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        return self.ret[index], self.ret2[index]

    def target_func(self, t, temp):
        return torch.tensor([math.sin(t * temp), math.sin(2 * t * temp)])

    def make_data(self, start, data_num, step, noise, target_func):
        ret = []
        ret2 = []
        t_list = [
            t * step for t in range(int(start / step), int(start / step + data_num))
        ]
        target_size = len(t_list), len(self.target_func(0, 0))
        rand_list = torch.randn(target_size) * noise
        for t, rand in zip(t_list, rand_list):
            temp = 1 if t < 314 * step else 2
            x = list(target_func(t, temp) + rand)
            x2 = list(target_func(t + step, temp))
            ret.append(x)
            ret2.append(x2)
        x = torch.tensor(ret).view(-1)
        return torch.tensor(ret).view(-1, 2), torch.tensor(ret2).view(-1, 2)
