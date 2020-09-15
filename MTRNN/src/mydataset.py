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
        super(CsvDataSet,self).__init__()
        self._paths = [str(p) for p in Path(dir_path).glob("./*/*.csv")]
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


class CustomDataSet(torch.utils.data.Dataset):
    def __init__(self, dir_path, tactile_frame_num):
        super().__init__()
        self._paths = [str(p) for p in Path(dir_path).glob("./*/*.csv")]
        self._paths.sort()
        self._datas = []
        for path in self._paths:
            df = pd.read_csv(path)
            self._datas.append(df)
        self._len = len(self._datas)
        self._datas = np.array(self._datas)
        self._motor = self._datas[:, :, :14]
        tactile = self._datas[:, :, 14:30]
        self._img = self._datas[:, :, 30:]
        zero = np.zeros(shape=(tactile_frame_num - 1, tactile.shape[2]))
        # for i in tactile:
        #     print(i.shape)
        tactile_with0 = [np.concatenate([zero, i], axis=0) for i in tactile]
        # tactile_with0 = np.concatenate(tactile_with0, axis=0)
        tactile_with0 = np.array(tactile_with0)
        tactile5 = [
            [tactile_with0[:, i : i + tactile_frame_num]]
            for i in range(tactile[0].shape[0])
        ]
        tactile5 = np.array(tactile5)
        self._tactile = np.transpose(tactile5, (2, 0, 1, 3, 4))
        # self._tactile = np.array(self._tactile)
        self._motor = torch.from_numpy(self._motor).float()
        self._img = torch.from_numpy(self._img).float()
        self._tactile = torch.from_numpy(self._tactile).float()

    def __getitem__(self, index):
        # print(self._datas[index][1:].shape)
        return (
            [
                self._motor[index][0:-1],
                self._tactile[index][0:-1],
                self._img[index][0:-1],
            ],
            self._datas[index][1:],
        )

    def __len__(self):
        return self._len
