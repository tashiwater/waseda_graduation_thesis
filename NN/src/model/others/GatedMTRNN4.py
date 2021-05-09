#!/usr/bin/env python
# coding:utf-8


import torch
from .MTRNN import MTRNN
from .cell import Cell


class GatedMTRNN(torch.nn.Module):  # [TODO]cannot use GPU now
    def __init__(
        self,
        layer_size={"in": 1, "out": 1, "io": 3, "cf": 4, "cs": 5},
        tau={"tau_io": 2, "tau_cf": 5.0, "tau_cs": 70.0},
        open_rate=1,
    ):
        super(GatedMTRNN, self).__init__()
        self.open_rate = open_rate
        self.mtrnn = MTRNN(layer_size, tau, 1, torch.nn.Tanh())
        # self._position_dims = 7  # layer_size["out"]
        # sensor_dims = layer_size["in"] - self._position_dims
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(layer_size["in"], 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 3),
            torch.nn.Sigmoid(),
        )

    def init_state(self, batch_size):
        self.mtrnn.init_state(batch_size)

    def forward(self, x):
        if self.mtrnn.last_output is not None:  # start val is not changed by open_rate
            x = self.open_rate * x + self.mtrnn.last_output * (1 - self.open_rate)
        self.attention_map = self.attention(x)
        # a = self.attention_map[:, 0].unsqueeze(1).repeat(1, 7)
        position = x[:, :7]  # * self.attention_map[:, 0:1].repeat(1, 7)
        torque = x[:, 7:14] * self.attention_map[:, 0].unsqueeze(1).repeat(1, 7)
        tactile = x[:, 14:26] * self.attention_map[:, 1].unsqueeze(1).repeat(1, 12)
        img = x[:, 26:] * self.attention_map[:, 2].unsqueeze(1).repeat(1, 15)
        # temp = x * self.attention_map
        temp = torch.cat([position, torque, tactile, img], axis=1)
        return self.mtrnn(temp)
