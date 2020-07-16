#!/usr/bin/env python
# coding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MTRNN(nn.Module):  # [TODO]cannot use GPU now
    def __init__(
        self,
        in_size,
        out_size,
        c_size={"io": 3, "cf": 4, "cs": 5},
        tau={"tau_io": 2, "tau_cf": 5.0, "tau_cs": 70.0},
    ):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.c_size = c_size
        self.tau = tau
        self.i2io = nn.Linear(self.in_size, self.c_size["io"])
        self.io2o = nn.Linear(self.c_size["io"], self.out_size)
        self.io2io = nn.Linear(self.c_size["io"], self.c_size["io"])
        self.io2cf = nn.Linear(self.c_size["io"], self.c_size["cf"])
        self.cf2io = nn.Linear(self.c_size["cf"], self.c_size["io"])
        self.cf2cs = nn.Linear(self.c_size["cf"], self.c_size["cs"])
        self.cf2cf = nn.Linear(self.c_size["cf"], self.c_size["cf"])
        self.cs2cf = nn.Linear(self.c_size["cs"], self.c_size["cf"])
        self.cs2cs = nn.Linear(self.c_size["cs"], self.c_size["cs"])
        self.activate = torch.nn.Tanh()

    def init_state(self, batch_size):
        # self.io_state = torch.zeros(size=(batch_size, self.c_size["io"]))
        # self.cf_state = torch.zeros(size=(batch_size, self.c_size["cf"]))
        # self.cs_state = torch.zeros(size=(batch_size, self.c_size["cs"]))
        self.io_state = torch.full(size=(batch_size, self.c_size["io"]), fill_value=0.5)
        self.cf_state = torch.full(size=(batch_size, self.c_size["cf"]), fill_value=0.5)
        self.cs_state = torch.full(size=(batch_size, self.c_size["cs"]), fill_value=0.5)

    def _next_state(self, previous, new, tau):
        connected = torch.stack(new)
        new_summed = connected.sum(dim=0)
        ret = (1 - 1 / tau) * previous + new_summed / tau
        return self.activate(ret)

    def forward(self, x):  # x.shape(batch,x)
        new_io_state = self._next_state(
            previous=self.io_state,
            new=[self.io2io(self.io_state), self.cf2io(self.cf_state), self.i2io(x),],
            tau=self.tau["tau_io"],
        )
        new_cf_state = self._next_state(
            previous=self.cf_state,
            new=[
                self.cf2cf(self.cf_state),
                self.cs2cf(self.cs_state),
                self.io2cf(self.io_state),
            ],
            tau=self.tau["tau_cf"],
        )
        new_cs_state = self._next_state(
            previous=self.cs_state,
            new=[self.cs2cs(self.cs_state), self.cf2cs(self.cf_state),],
            tau=self.tau["tau_cs"],
        )
        self.io_state = new_io_state
        self.cf_state = new_cf_state
        self.cs_state = new_cs_state
        y = self.activate(self.io2o(self.io_state))
        return y
