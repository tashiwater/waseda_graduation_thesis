#!/usr/bin/env python3
# coding: utf-8
import os
from mydataset import CsvDataSet as MyDataSet
from MTRNN import MTRNN
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = CURRENT_DIR + "/../data/"
RESULT_DIR = DATA_DIR + "result/"
# RESULT2_DIR = DATA_DIR + "result_correct/"
MODEL_DIR = DATA_DIR + "model_MTRNN_from_csv/"
VALIDATE_PATH = DATA_DIR + "test"

dataset = MyDataSet(VALIDATE_PATH)
in_size = dataset[0][0].shape[2]
net = MTRNN(
    in_size=in_size,
    out_size=in_size,
    batch_size=1,
    c_size={"io": 34, "cf": 160, "cs": 13},
    tau={"tau_io": 2, "tau_cf": 5, "tau_cs": 50},
)

### modelをロード
model_path = MODEL_DIR + "20200710_135240_6500.pth"
checkpoint = torch.load(model_path)
net.load_state_dict(checkpoint["model"])

testloader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=False, num_workers=4,
)

criterion = torch.nn.MSELoss()
net.eval()
calc_num = 0
sum_loss = 0
for i, (inputs, labels) in enumerate(testloader):
    for inputs, labels in zip(inputs, labels):
        net.init_state()
        outputs = torch.empty_like(labels)
        for t, (data, label) in enumerate(zip(inputs, labels)):
            close_rate = 0
            if t > 1:
                data = output.cpu() * close_rate + data * (1 - close_rate)
            output = net(data)
            loss = criterion(output, label)
            sum_loss += loss.item()
            calc_num += 1
            outputs[t] = output
    mean_loss = sum_loss / calc_num
    print("meanloss={}".format(mean_loss))
    np_output = outputs.view(-1, dataset[0][0].shape[2]).detach().numpy()
    np.savetxt(
        RESULT_DIR + "{}.csv".format(i), np_output, delimiter=",",
    )

# t_list, t_list2 = get_t(1, 600, 0.02)
# fig = plt.figure()
# # print(outputs.view(-1).detach().numpy())
# # plt.scatter(t_list2, outputs.detach().numpy(), c="red", label="pred")
# plt.scatter(t_list2, labels.view(-1), c="blue", label="y=sin(x)")
# plt.legend()
# # fig.savefig("temp.png")
# plt.show()

# fig = plt.figure()
# pred_x_t = [one_data[0][0] for one_data in outputs.detach().numpy()]
# pred_y_t = [one_data[0][1] for one_data in outputs.detach().numpy()]
# test_x_t = [one_data[0][0] for one_data in labels.detach().numpy()]
# test_y_t = [one_data[0][1] for one_data in labels.detach().numpy()]
# plt.scatter(pred_x_t, pred_y_t, c="red", label="pred")
# plt.scatter(test_x_t, test_y_t, c="blue", label="test")
# plt.legend()
# # fig.savefig("temp.png")
# plt.show()
