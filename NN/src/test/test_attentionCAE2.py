#!/usr/bin/env python3
# coding: utf-8
import os
import sys
import torch
import torchvision
from PIL import Image, ImageChops

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dataset.dataset_CAE import MyDataSet
from model.AttentionCAE2 import AttentionCAE as Net

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = CURRENT_DIR + "/../../data/CAE/"
DATA_PATH = DATA_DIR + "validate"
RESULT_DIR = DATA_DIR + "result/"
# CORRECT_DIR = DATA_DIR + "result_correct/"
MODEL_BASE = "/media/hdd_1tb/model/"
MODEL_BASE = CURRENT_DIR + "/../../../../model/"
MODEL_DIR = MODEL_BASE + "AttentionCAE2/theta0/"

net = Net()

### modelをロード
model_path = MODEL_DIR + "20201020_041027_6000.pth"
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
for i, (inputs, labels) in enumerate(testloader):
    inputs = inputs.to(device)
    labels = [labels[k].to(device) for k in range(2)]
    outputs = net.encoder(inputs)
    val, class_num = torch.max(outputs[1], 1)
    outputs = net.decoder(outputs[0])
    loss = criterion(outputs, labels[0])
    print(loss.item())
    # print(torch.min(inputs))
    # for j, img in enumerate(inputs.cpu()):
    #     MyDataSet.save_img(img, CORRECT_DIR + "{}_{}.png".format(i, j))
    # torchvision.utils.save_image(img, CORRECT_DIR + "{}_{}.png".format(i, j))

    for j, img in enumerate(outputs.cpu()):
        MyDataSet.save_img(
            img, RESULT_DIR + "{}_{}class{}.png".format(i, j, class_num[j])
        )
        # torchvision.utils.save_image(img, RESULT_DIR + "{}_{}.png".format(i, j))
    for j, (img, mask) in enumerate(zip(inputs.cpu(), net.attention_map.cpu())):
        rgb = torchvision.transforms.functional.to_pil_image(img, "RGB")

        gray = torchvision.transforms.functional.to_pil_image(mask, "L")
        gray = gray.resize((128, 96), Image.NEAREST).convert("RGB")
        add_img = ImageChops.multiply(rgb, gray)
        add_img.save(RESULT_DIR + "{}_{}_attention.png".format(i, j))
###model output
# torch.onnx.export(net, inputs, RESULT_DIR + "model.onnx", verbose=True)
# print(outputs.size())
# for i, img in enumerate(inputs.cpu()):
#     torchvision.utils.save_image(img, RESULT2_DIR + "{}.png".format(i))

# for i, img in enumerate(outputs.cpu()):
#     torchvision.utils.save_image(img, RESULT2_DIR + "{}.png".format(i))
