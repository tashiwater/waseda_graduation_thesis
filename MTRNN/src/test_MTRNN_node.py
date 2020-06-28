#!/usr/bin/env python3
# coding: utf-8
import os
from mydataset import EasyDataSet as MyDataSet
from MTRNN import MTRNNCell, CustomRNN
import torch
import torchvision
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = CURRENT_DIR + "/../data/"
# DATA_PATH = DATA_DIR + "validate"
# RESULT_DIR = DATA_DIR + "result/"
# RESULT2_DIR = DATA_DIR + "result_correct/"
MODEL_DIR = DATA_DIR + "model_MTRNN/"

net = MTRNNCell()

### modelをロード
model_path = MODEL_DIR + "20200628_135805_100finish.pth"
net.load_state_dict(torch.load(model_path))

dataset = MyDataSet(1, 600, 0.02, batch_num=1)
testloader = torch.utils.data.DataLoader(
    dataset, batch_size=len(dataset), shuffle=False, num_workers=4,
)

device = torch.device("cuda:0")
criterion = torch.nn.MSELoss()
net = net.to(device)
net.eval()
calc_num = 0
sum_loss = 0
for i, (inputs, labels) in enumerate(testloader):
    for inputs, labels in zip(inputs, labels):
        hidden_state = net._get_initial_hidden_states(inputs[0].size(0))
        outputs = torch.empty_like(labels)
        for t, (data, label) in enumerate(zip(inputs, labels)):
            data, label, = data.to(device), label.to(device)
            hidden_state = [hidden.to(device) for hidden in hidden_state]
            output, hidden_state = net(data, hidden_state)
            loss = criterion(output, label)
            sum_loss += loss.item()
            calc_num += 1
            outputs[t] = output
    mean_loss = sum_loss / calc_num
    print("meanloss={}".format(mean_loss))


def get_t(start, data_num, step):
    t_list = [t * step for t in range(int(start / step), int(start / step + data_num))]
    t_list2 = [
        t * step + step for t in range(int(start / step), int(start / step + data_num))
    ]
    return t_list, t_list2


t_list, t_list2 = get_t(1, 600, 0.02)
fig = plt.figure()
# print(outputs.view(-1).detach().numpy())
plt.scatter(t_list2, outputs.detach().numpy(), c="red", label="pred")
plt.scatter(t_list2, labels.view(-1), c="blue", label="y=sin(x)")
plt.legend()
fig.savefig("temp.png")
plt.show()
# for j, img in enumerate(inputs.cpu()):
#     torchvision.utils.save_image(img, RESULT2_DIR + "{}_{}.png".format(i, j))

# for j, img in enumerate(outputs.cpu()):
#     torchvision.utils.save_image(img, RESULT_DIR + "{}_{}.png".format(i, j))

###model output
# torch.onnx.export(net, inputs, RESULT_DIR + "model.onnx", verbose=True)
# print(outputs.size())
# for i, img in enumerate(inputs.cpu()):
#     torchvision.utils.save_image(img, RESULT2_DIR + "{}.png".format(i))

# for i, img in enumerate(outputs.cpu()):
#     torchvision.utils.save_image(img, RESULT2_DIR + "{}.png".format(i))
