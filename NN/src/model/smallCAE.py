#!/usr/bin/env python3
# coding: utf-8
import torch
from torchsummary import summary
from cell import Cell


class ToImg(torch.nn.Module):
    def forward(self, x):
        n, _ = x.shape
        return x.reshape(n, 64, 12, 16)


class CAE(torch.nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        hidden_dim = 15  # 20
        self.encoder = torch.nn.Sequential(
            # 128*96*3
            Cell(3, 16),  # 64*48
            Cell(16, 32),  # 32*i24
            Cell(32, 64),  # 16*12
            torch.nn.Flatten(),
            Cell(16 * 12 * 64, 254, mode="linear"),
            Cell(254, hidden_dim, activate="sigmoid", mode="linear"),
        )
        self.decoder = torch.nn.Sequential(
            Cell(hidden_dim, 254, mode="linear"),
            Cell(254, 16 * 12 * 64, mode="linear"),
            ToImg(),
            Cell(64, 32, mode="conv_trans"),
            Cell(32, 16, mode="conv_trans"),
            Cell(16, 3, mode="conv_trans", activate="relu", on_batchnorm=False),
        )

    def forward(self, x):
        hidden = self.encoder(x)
        x = self.decoder(hidden)
        return x


if __name__ == "__main__":
    net = CAE()
    # device = torch.device("cuda:0")
    # net = net.to(device)
    summary(net, (3, 96, 128), device="cpu")
