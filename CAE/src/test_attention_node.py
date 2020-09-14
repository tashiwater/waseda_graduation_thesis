#!/usr/bin/env python3
# coding: utf-8
import os
from mydataset import MyDataSetForAttention as MyDataSet
from CAE import CAEwithAttention as Net
import torch
import torchvision
from PIL import Image, ImageChops

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = CURRENT_DIR + "/../data/"
DATA_PATH = DATA_DIR + "test_attention"
RESULT_DIR = DATA_DIR + "result/"
CORRECT_DIR = DATA_DIR + "result_correct/"
MODEL_DIR = DATA_DIR + "model_CAE20/"

net = Net()

### modelをロード
model_path = MODEL_DIR + "attention0001/20200911_212809_2200.pth"
checkpoint = torch.load(model_path)
net.load_state_dict(checkpoint["model"])
torch.save(
    net.state_dict(),
    MODEL_DIR + "attention0001/model.pth",
    _use_new_zipfile_serialization=False,
)

dataset = MyDataSet(DATA_PATH, noise=0)
testloader = torch.utils.data.DataLoader(
    dataset, batch_size=500, shuffle=False, num_workers=4,
)

device = torch.device("cuda:0")
criterion = torch.nn.MSELoss()
net = net.to(device)
net.eval()
for i, (inputs, labels) in enumerate(testloader):
    inputs = inputs.to(device)  # , labels.to(device)
    outputs = net.encoder(inputs)
    val, class_num = torch.max(outputs[1], 1)
    outputs = net.decoder(outputs[0])
    attention_maps = net.attention_map
    # loss = criterion(outputs, labels)
    # print(loss.item())
    # print(torch.min(inputs))
    inputs = inputs.cpu()
    for j, img in enumerate(inputs):
        MyDataSet.save_img(img, CORRECT_DIR + "{}_{}.png".format(i, j))
    # torchvision.utils.save_image(img, CORRECT_DIR + "{}_{}.png".format(i, j))
    outputs = outputs.cpu()
    for j, img in enumerate(outputs):
        MyDataSet.save_img(
            img, RESULT_DIR + "{}_{}_class{}.png".format(i, j, class_num[j])
        )

    for j, (img, mask) in enumerate(zip(inputs, attention_maps.cpu())):
        rgb = torchvision.transforms.functional.to_pil_image(img, "RGB")

        gray = torchvision.transforms.functional.to_pil_image(mask, "L")
        gray = gray.resize((128, 96), Image.NEAREST).convert("RGB")
        add_img = ImageChops.multiply(rgb, gray)
        add_img.save(RESULT_DIR + "{}_{}_attention.png".format(i, j))
        gray.save(RESULT_DIR + "{}_{}gray.png".format(i, j))

###model output
# torch.onnx.export(net, inputs, RESULT_DIR + "model.onnx", verbose=True)
# print(outputs.size())
# for i, img in enumerate(inputs.cpu()):
#     torchvision.utils.save_image(img, RESULT2_DIR + "{}.png".format(i))

# for i, img in enumerate(outputs.cpu()):
#     torchvision.utils.save_image(img, RESULT2_DIR + "{}.png".format(i))
