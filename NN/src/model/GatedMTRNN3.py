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
        self.mtrnn = MTRNN(layer_size, tau, 1)
        # self._position_dims = 7  # layer_size["out"]
        # sensor_dims = layer_size["in"] - self._position_dims
        self.attention = torch.nn.Sequential(
            Cell(
                layer_size["in"],
                100,
                activate="sigmoid",
                mode="linear",
                on_batchnorm=False,
            ),
            Cell(100, 100, activate="sigmoid", mode="linear", on_batchnorm=False),
            # Cell(100, 100, activate="sigmoid", mode="linear", on_batchnorm=False),
            # Cell(100, 100, activate="sigmoid", mode="linear", on_batchnorm=False),
            Cell(
                100,
                layer_size["in"],
                activate="softmax",
                mode="linear",
                on_batchnorm=False,
            ),
        )

    def init_state(self, batch_size):
        self.mtrnn.init_state(batch_size)

    def forward(self, x):
        # position = x[:, : self._position_dims]
        if self.mtrnn.last_output is not None:  # start val is not changed by open_rate
            x = self.open_rate * x + self.mtrnn.last_output * (1 - self.open_rate)
        # sensors = x[:, self._position_dims :]
        self.attention_map = self.attention(x)
        x = x * self.attention_map
        # Sx = torch.cat([position, sensors], axis=1)
        return self.mtrnn(x)
