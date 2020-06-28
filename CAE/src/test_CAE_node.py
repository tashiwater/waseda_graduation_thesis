#!/usr/bin/env python3
# coding: utf-8
import os
from mydataset import MyDataSetForCAE as MyDataSet
from CAE import Net
from train_net_predict import TrainNet
import torch
import torchvision

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = CURRENT_DIR + "/../data/"
DATA_PATH = DATA_DIR + "validate"
RESULT_DIR = DATA_DIR + "result/"
RESULT2_DIR = DATA_DIR + "result_correct/"
MODEL_DIR = DATA_DIR + "model_CAE20/"

net = Net()

### modelをロード
model_path = MODEL_DIR + "6_22/20200623_093348_200finish.pth"
net.load_state_dict(torch.load(model_path))

dataset = MyDataSet(DATA_PATH)
testloader = torch.utils.data.DataLoader(
    dataset, batch_size=len(dataset), shuffle=False, num_workers=4,
)

device = torch.device("cuda:0")
criterion = torch.nn.MSELoss()
net = net.to(device)
net.eval()
for i, (inputs, labels) in enumerate(testloader):
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    print(loss.item())
    for j, img in enumerate(inputs.cpu()):
        torchvision.utils.save_image(img, RESULT2_DIR + "{}_{}.png".format(i, j))

    for j, img in enumerate(outputs.cpu()):
        torchvision.utils.save_image(img, RESULT_DIR + "{}_{}.png".format(i, j))

###model output
# torch.onnx.export(net, inputs, RESULT_DIR + "model.onnx", verbose=True)
# print(outputs.size())
# for i, img in enumerate(inputs.cpu()):
#     torchvision.utils.save_image(img, RESULT2_DIR + "{}.png".format(i))

# for i, img in enumerate(outputs.cpu()):
#     torchvision.utils.save_image(img, RESULT2_DIR + "{}.png".format(i))
