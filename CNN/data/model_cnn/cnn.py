#!/usr/bin/env python3
# coding: utf-8
import torch


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, stride=2)

        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        # self.conv3 = nn.Conv2d(16, 24, 5)

        self.fc1 = torch.nn.Linear(29 * 21 * 16, 1000)
        self.fc2 = torch.nn.Linear(1000, 120)
        self.fc3 = torch.nn.Linear(120, 60)
        self.fc4 = torch.nn.Linear(60, 4)  # 3 classes

    def forward(self, x):
        # x:128*96*3
        x = self.conv1(x)  # ->124*92*6
        x = self.relu(x)
        x = self.pool(x)  # ->62*46*6
        x = self.conv2(x)  # ->58*42*16
        x = self.relu(x)
        x = self.pool(x)  # ->29*21*16
        # x = self.conv3(x) #->
        # x = self.relu(x)
        # x = self.pool(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x
