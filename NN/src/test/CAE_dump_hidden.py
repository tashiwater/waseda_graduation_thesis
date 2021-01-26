#!/usr/bin/env python3
# coding: utf-8
import os
import sys
import torch
from pathlib import Path
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dataset.dataset_CAE import OneDataSet
from model.CAE import CAE as Net

#
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
my_dir = "CAE/0106/all/"
DATA_DIR = CURRENT_DIR + "/../../data/" + my_dir
IMAGE_PATH = DATA_DIR + "all/"
MODEL_BASE = "/media/user/ボリューム/model/"
model_path = (
    CURRENT_DIR + "/../../../../model/" + my_dir + "20210113_221042_5000finish.pth"
)

HIDDEN_DIR = CURRENT_DIR + "/../../../preprocess/data/connect_input/image_feature/"

net = Net()

### modelをロード
# paths = [str(p) for p in Path(model_path).glob("./*finish.pth")]
# if len(paths) != 1:
#     raise FileExistsError("there are {} finish.pth".format(len(paths)))
# model_path = paths[0]
print(model_path)
checkpoint = torch.load(model_path)
net.load_state_dict(checkpoint["model"])


device = torch.device("cuda:0")
# criterion = torch.nn.MSELoss()
net = net.to(device)
net.eval()
# get img dirs
img_dirs = [str(p) for p in Path(IMAGE_PATH).glob("./*/*")]
img_dirs.sort()

for j, img_dir in enumerate(img_dirs):  # deal with each file
    dataset = OneDataSet(img_dir, img_size=(128, 96), is_test=True, dsize=5)
    testloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=len(dataset),
        shuffle=False,
        num_workers=4,
    )

    for i, (inputs, labels) in enumerate(testloader):
        inputs = inputs.to(device)
        outputs = net.encoder(inputs)
        # imgs = net.decoder(outputs)
        np.savetxt(
            HIDDEN_DIR + "{:03}.csv".format(j),
            outputs.to("cpu").detach().numpy(),
            delimiter=",",
        )
        print(HIDDEN_DIR + "{:03}.csv".format(j))
