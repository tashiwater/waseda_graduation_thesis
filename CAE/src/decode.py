#!/usr/bin/env python3
# coding: utf-8
import os
from mydataset import MyDataSetForCAE as MyDataSet
from CAE import Net
from train_net_predict import TrainNet
import torch
import torchvision
import numpy as np
from pathlib import Path

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = CURRENT_DIR + "/../data/"
MODEL_DIR = DATA_DIR + "model_CAE20/"
RESULT_DIR = DATA_DIR + "result/"

net = Net()

### modelをロード
model_path = MODEL_DIR + "use_all/20200830_115126_100.pth"
checkpoint = torch.load(model_path)
net.load_state_dict(checkpoint["model"])

device = torch.device("cuda:0")
# criterion = torch.nn.MSELoss()
net = net.to(device)
net.eval()
# get img dirs
size = (1, 20)
for j in range(100):
    fill_value = 0.01 * j
    inputs = torch.full(size=size, fill_value=fill_value)
    inputs = inputs.to(device)
    imgs = net.decoder(inputs)
    for i, img in enumerate(imgs):
        torchvision.utils.save_image(img, RESULT_DIR + "{}.png".format(j))
