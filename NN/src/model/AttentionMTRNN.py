#!/usr/bin/env python
# coding:utf-8


import torch
from .MTRNN import MTRNN
from .cell import Cell


class AttentionMTRNN(torch.nn.Module):  # [TODO]cannot use GPU now
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
            torch.nn.Linear(layer_size["cs"], 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 2),
            torch.nn.Sigmoid(),
        )

    def init_state(self, batch_size):
        self.mtrnn.init_state(batch_size)

    def forward(self, x):
        if self.mtrnn.last_output is not None:  # start val is not changed by open_rate
            x = self.open_rate * x + self.mtrnn.last_output * (1 - self.open_rate)
        temp = self.attention(self.mtrnn.cs_state)
        self.attention_map = torch.ones_like(x)
        self.attention_map[:, 7:30] = temp[:, 0].unsqueeze(1).repeat(1, 23)
        self.attention_map[:, 30:] = temp[:, 1].unsqueeze(1).repeat(1, 15)
        return self.mtrnn(x * self.attention_map)
