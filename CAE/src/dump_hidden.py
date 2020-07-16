#!/usr/bin/env python3
# coding: utf-8
import os
from mydataset import MyImgSet as MyDataSet
from CAE import Net
from train_net_predict import TrainNet
import torch
import torchvision
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = CURRENT_DIR + "/../data/"
DATA_PATH = DATA_DIR + "image/"
RESULT_DIR = DATA_DIR + "result/"
MODEL_DIR = DATA_DIR + "model_CAE20/"
HIDDEN_DIR = DATA_DIR + "hidden/"

net = Net()

### modelをロード
model_path = MODEL_DIR + "6_22/20200623_093348_200finish.pth"
net.load_state_dict(torch.load(model_path))


device = torch.device("cuda:0")
# criterion = torch.nn.MSELoss()
net = net.to(device)
net.eval()

for j in range(1, 5):
    dataset = MyDataSet(DATA_PATH + "image{}/".format(j))
    testloader = torch.utils.data.DataLoader(
        dataset, batch_size=215, shuffle=False, num_workers=4,
    )

    for i, (inputs, labels) in enumerate(testloader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net.encoder(inputs)
        # imgs = net.decoder(outputs)
        np.savetxt(
            HIDDEN_DIR + "image{}/{}.csv".format(j, i + 1),
            outputs.to("cpu").detach().numpy(),
            delimiter=",",
        )
        # for j, img in enumerate(imgs.cpu()):
        #     torchvision.utils.save_image(img, RESULT_DIR + "{}_{}.png".format(i, j))
