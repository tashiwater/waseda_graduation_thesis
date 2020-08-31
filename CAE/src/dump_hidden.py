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
IMAGE_PATH = DATA_DIR + "image/"
RESULT_DIR = DATA_DIR + "result/"
MODEL_DIR = DATA_DIR + "model_CAE20/"
HIDDEN_DIR = DATA_DIR + "image_feature/"

net = Net()

### modelをロード
model_path = MODEL_DIR + "20200827_225616_200.pth"
net.load_state_dict(torch.load(model_path))


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
        dataset, batch_size=len(dataset), shuffle=False, num_workers=4,
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
        # for j, img in enumerate(imgs.cpu()):
        #     torchvision.utils.save_image(img, RESULT_DIR + "{}_{}.png".format(i, j))
