#!/usr/bin/env python3
# coding: utf-8
import torch
import torchvision
from pathlib import Path
from PIL import Image
import math

"""IMAGE DATASET"""


class EasyDataSet(torch.utils.data.Dataset):
    def __init__(self, start, data_num, step, batch_num):
        x_datas, y_datas = [], []
        for i in range(batch_num):
            x, y = self.make_data(start + i, data_num, step)
            x_datas.append(x)
            y_datas.append(y)

        self.ret, self.ret2 = x_datas, y_datas
        # torch.tensor(x_datas), torch.tensor(y_datas)
        self._len = len(self.ret)

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        return self.ret[index], self.ret2[index]

    def target_func(self, t):
        return math.sin(t)

    def make_data(self, start, data_num, step):
        ret = []
        ret2 = []
        t_list = [
            t * step for t in range(int(start / step), int(start / step + data_num))
        ]
        for t in t_list:
            ret.append(self.target_func(t))
            ret2.append(self.target_func(t + step))
        # x = torch.tensor(t_list).view(-1, 1, 1)
        return torch.tensor(ret).view(-1, 1, 1), torch.tensor(ret2).view(-1, 1, 1)


class mkDataSet(torch.utils.data.Dataset):
    def __init__(self, data_size, data_length=50, freq=60.0):
        """
        params
        data_size : データセットのサイズ
        data_length : 各データの時系列長
        freq : 周波数
        returns
        train_x : トレーニングデータ（t=1,2,...,size-1の値)
        train_t : トレーニングデータのラベル（t=sizeの値）
        """
        train_x = []
        train_t = []

        for offset in range(data_size):
            train_x.append(
                [
                    [math.sin(2 * math.pi * (offset + i) / freq)]
                    for i in range(data_length)
                ]
            )
            train_t.append([math.sin(2 * math.pi * (offset + data_length) / freq)])

        self.ret, self.ret2 = torch.tensor(train_x), torch.tensor(train_t)
        self._len = len(self.ret)

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        return self.ret[index], self.ret2[index]
