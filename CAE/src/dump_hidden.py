#!/usr/bin/env python3
# coding: utf-8
import os
from mydataset import MyDataSetForCAE as MyDataSet
from CAE import CAE as Net
import torch
import torchvision
import numpy as np
from pathlib import Path

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = CURRENT_DIR + "/../data/"
IMAGE_PATH = DATA_DIR + "all/"
MODEL_BASE = CURRENT_DIR + "/../../../../model/"
MODEL_DIR = MODEL_BASE + "CAE/newcam/"
HIDDEN_DIR = "/home/assimilation/TAKUMI_SHIMIZU/waseda_graduation_thesis/preprocess/data/connect_input/image_feature/"

net = Net()

### modelをロード
model_path = MODEL_DIR + "best/20201005_170823_2000.pth"
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
    dataset = MyDataSet(img_dir, jpg_path="./*.jpg")
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
            outputs[0].to("cpu").detach().numpy(),
            delimiter=",",
        )
        # for j, img in enumerate(imgs.cpu()):
        #     torchvision.utils.save_image(img, RESULT_DIR + "{}_{}.png".format(i, j))
