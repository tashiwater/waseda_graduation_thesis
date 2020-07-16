#!/usr/bin/env python3
# coding: utf-8
import os
from mydataset import CsvDataSet as MyDataSet
from MTRNN import MTRNNCell, CustomRNN
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
net = MTRNNCell(
    input_output_size=dataset[0][0].shape[2],
    io_hidden_size=34,
    fast_hidden_size=160,
    slow_hidden_size=13,
    tau_input_output=2,
    tau_fast_hidden=5,
    tau_slow_hidden=50,
)

### modelをロード
model_path = MODEL_DIR + "20200705_175644_150.pth"
net.load_state_dict(torch.load(model_path))

testloader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=False, num_workers=4,
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
            close_rate = 0
            if t > 1:
                data = output.cpu() * close_rate + data * (1 - close_rate)
            data, label, = data.to(device), label.to(device)
            hidden_state = [hidden.to(device) for hidden in hidden_state]
            output, hidden_state = net(data, hidden_state)
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
