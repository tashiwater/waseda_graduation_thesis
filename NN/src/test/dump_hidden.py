#!/usr/bin/env python3
# coding: utf-8
import os
import sys
import torch
from pathlib import Path
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dataset.dataset_CAE import OneDataSet
from model.AttentionCAE import AttentionCAE as Net

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = CURRENT_DIR + "/../../data/CAE/"
IMAGE_PATH = DATA_DIR + "all/"
MODEL_BASE = "/media/hdd_1tb/model/"
MODEL_BASE = CURRENT_DIR + "/../../../../model/"
MODEL_DIR = MODEL_BASE + "AttentionCAE/"

HIDDEN_DIR = CURRENT_DIR + "/../../../preprocess/data/connect_input/image_feature/"

net = Net()

### modelをロード
model_path = MODEL_DIR + "0/20201022_194841_2500.pth"
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
