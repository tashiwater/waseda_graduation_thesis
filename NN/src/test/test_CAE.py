#!/usr/bin/env python3
# coding: utf-8
import os
import sys
import torch
import torchvision

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dataset.dataset_CAE import MyDataSet
from model.CAE import CAE as Net

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = CURRENT_DIR + "/../../data/CAE/"
DATA_PATH = DATA_DIR + "test"
RESULT_DIR = DATA_DIR + "result/"
# CORRECT_DIR = DATA_DIR + "result_correct/"
MODEL_BASE = CURRENT_DIR + "/../../../../model/"
MODEL_DIR = MODEL_BASE + "CAE/theta0_mix/"

net = Net()

### modelをロード
model_path = MODEL_DIR + "20201103_161859_500.pth"
checkpoint = torch.load(model_path)
net.load_state_dict(checkpoint["model"])

dataset = MyDataSet(DATA_PATH, img_size=(128, 96), is_test=True, dsize=5)
testloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=500,
    shuffle=False,
    num_workers=4,
)

device = torch.device("cuda:0")
criterion = torch.nn.MSELoss()
net = net.to(device)
net.eval()
count = 1
for i, (inputs, labels) in enumerate(testloader):
    inputs = inputs.to(device)
    labels = [labels[i].to(device) for i in range(2)]
    outputs = net(inputs)
    loss = criterion(outputs, labels[0])
    print(loss.item())
    # print(torch.min(inputs))
    # for j, img in enumerate(inputs.cpu()):
    #     MyDataSet.save_img(img, CORRECT_DIR + "{}_{}.png".format(i, j))
    # torchvision.utils.save_image(img, CORRECT_DIR + "{}_{}.png".format(i, j))

    for j, img in enumerate(outputs.cpu()):
        MyDataSet.save_img(img, RESULT_DIR + "{:03d}.png".format(count))
        count += 1
        # torchvision.utils.save_image(img, RESULT_DIR + "{}_{}.png".format(i, j))

###model output
# torch.onnx.export(net, inputs, RESULT_DIR + "model.onnx", verbose=True)
# print(outputs.size())
# for i, img in enumerate(inputs.cpu()):
#     torchvision.utils.save_image(img, RESULT2_DIR + "{}.png".format(i))

# for i, img in enumerate(outputs.cpu()):
#     torchvision.utils.save_image(img, RESULT2_DIR + "{}.png".format(i))
