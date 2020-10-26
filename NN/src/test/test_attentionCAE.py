#!/usr/bin/env python3
# coding: utf-8
import os
import sys
import torch
import torchvision
from PIL import Image, ImageChops

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dataset.dataset_CAE import MyDataSet
from model.AttentionCAE import AttentionCAE as Net

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = CURRENT_DIR + "/../../data/CAE/"
DATA_PATH = DATA_DIR + "test"
RESULT_DIR = DATA_DIR + "result/"
# CORRECT_DIR = DATA_DIR + "result_correct/"
MODEL_BASE = "/media/hdd_1tb/model/"
# MODEL_BASE = CURRENT_DIR + "/../../../../model/"
MODEL_DIR = MODEL_BASE + "AttentionCAE/"

net = Net()

### modelをロード
model_path = MODEL_DIR + "0/20201022_212259_5000.pth"
checkpoint = torch.load(model_path)
net.load_state_dict(checkpoint["model"])

dataset = MyDataSet(DATA_PATH, img_size=(128, 96), is_test=True, dsize=5)
testloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=200,
    shuffle=False,
    num_workers=4,
)

device = torch.device("cuda:0")
criterion = torch.nn.MSELoss()
net = net.to(device)
net.eval()

combined_imgs = []

for i, (inputs, labels) in enumerate(testloader):
    inputs = inputs.to(device)
    labels = [labels[k].to(device) for k in range(2)]
    outputs = net(inputs)
    loss = criterion(outputs, labels[0])
    print(loss.item())
    # print(torch.min(inputs))
    #     MyDataSet.save_img(img, CORRECT_DIR + "{}_{}.png".format(i, j))
    # torchvision.utils.save_image(img, CORRECT_DIR + "{}_{}.png".format(i, j))

    for j, img in enumerate(outputs.cpu()):
        MyDataSet.save_img(img, RESULT_DIR + "{}_{}.png".format(i, j))
        # torchvision.utils.save_image(img, RESULT_DIR + "{}_{}.png".format(i, j))
    for j, (img, mask) in enumerate(zip(inputs.cpu(), net.attention_map.cpu())):
        rgb = torchvision.transforms.functional.to_pil_image(img, "RGB")

        gray = torchvision.transforms.functional.to_pil_image(mask, "L")
        gray = gray.resize((128, 96), Image.NEAREST).convert("RGB")
        add_img = ImageChops.multiply(rgb, gray)
        add_img.save(RESULT_DIR + "{}_{}_attention.png".format(i, j))

        for img, output_img, mask in zip(
            inputs.cpu(), outputs.cpu(), net.attention_map.cpu()
        ):
            input_pil = torchvision.transforms.functional.to_pil_image(img)
            output_pil = torchvision.transforms.functional.to_pil_image(output_img)

            gray = torchvision.transforms.functional.to_pil_image(mask, "L")
            gray = gray.resize((128, 96), Image.NEAREST).convert("RGB")
            add_img = ImageChops.multiply(input_pil, gray)

            dst = Image.new(
                "RGB",
                (input_pil.width + output_pil.width + add_img.width, input_pil.height),
            )
            dst.paste(input_pil, (0, 0))
            # 二枚目なので左上のx座標はim1.widthを指定
            dst.paste(output_pil, (input_pil.width, 0))
            # 三枚目なので左上のx座標im1.width + im2.widthを指定
            dst.paste(add_img, (input_pil.width + output_pil.width, 0))
            combined_imgs.append(dst)

combined_imgs[0].save(
    "out.gif", save_all=True, append_images=combined_imgs[1:], loop=0, duration=100
)
###model output
# torch.onnx.export(net, inputs, RESULT_DIR + "model.onnx", verbose=True)
# print(outputs.size())
# for i, img in enumerate(inputs.cpu()):
#     torchvision.utils.save_image(img, RESULT2_DIR + "{}.png".format(i))

# for i, img in enumerate(outputs.cpu()):
#     torchvision.utils.save_image(img, RESULT2_DIR + "{}.png".format(i))
