#!/usr/bin/env python3
# coding: utf-8
import os
from mydataset import CsvDataSet as MyDataSet
from MTRNN import MTRNN
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = CURRENT_DIR + "/../data/"
RESULT_DIR = DATA_DIR + "result/"
# RESULT2_DIR = DATA_DIR + "result_correct/"
MODEL_DIR = DATA_DIR + "model/"
VALIDATE_PATH = DATA_DIR + "test"

dataset = MyDataSet(VALIDATE_PATH)
# dataset = MyDataSet(0, 600, 0.02, 1, 0.1)
in_size = dataset[0][0].shape[1]
net = MTRNN(
    layer_size={"in": in_size, "out": in_size, "io": 34, "cf": 160, "cs": 13},
    tau={"tau_io": 2, "tau_cf": 5, "tau_cs": 50},
    open_rate=0.3,
)
### modelをロード
model_path = MODEL_DIR + "20200719_181607_7000.pth"
checkpoint = torch.load(model_path)
net.load_state_dict(checkpoint["model"])
print(net)


def get_header(add_word):
    return (
        [add_word + "position{}".format(i) for i in range(7)]
        + [add_word + "torque{}".format(i) for i in range(7)]
        + [add_word + "tactile{}".format(i) for i in range(12)]
        + [add_word + "image{}".format(i) for i in range(20)]
    )


dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=False, num_workers=4,
)

criterion = torch.nn.MSELoss()
net.eval()
alltype_cs = []
for j, (one_batch_inputs, one_batch_labels) in enumerate(dataloader):
    inputs_transposed = one_batch_inputs.transpose(1, 0)
    labels_transposed = one_batch_labels.transpose(1, 0)
    net.init_state(inputs_transposed.shape[1])
    outputs = torch.zeros_like(labels_transposed)
    io_states = []
    cf_states = []
    cs_states = []
    for i, inputs_t in enumerate(inputs_transposed):
        outputs[i] = net(inputs_t)
        io_states.append(net.io_state.view(-1).detach().numpy())
        cf_states.append(net.cf_state.view(-1).detach().numpy())
        cs_states.append(net.cs_state.view(-1).detach().numpy())
    loss = criterion(outputs, labels_transposed)
    print("loss={}".format(loss.item()))
    alltype_cs += cs_states
    io_states = np.array(io_states)
    cf_states = np.array(cf_states)
    cs_states = np.array(cs_states)

    np_input = labels_transposed.view(-1, dataset[0][0].shape[1]).detach().numpy()
    np_output = outputs.view(-1, dataset[0][0].shape[1]).detach().numpy()
    cs_pca = PCA(n_components=2).fit_transform(cs_states)
    cf_pca = PCA(n_components=2).fit_transform(cf_states)
    connected_data = np.block(
        [np_input, np_output, cf_pca, cs_pca]  # , io_states, cf_states, cs_states
    )
    header = (
        # ["input{}".format(i) for i in range(np_input.shape[1])]
        # + ["output{}".format(i) for i in range(np_output.shape[1])]
        get_header("in ")
        + get_header("out ")
        + ["cf_pca{}".format(i) for i in range(cf_pca.shape[1])]
        + ["cs_pca{}".format(i) for i in range(cs_pca.shape[1])]
        # + ["io_states{}".format(i) for i in range(io_states.shape[1])]
        # + ["cf_states{}".format(i) for i in range(cf_states.shape[1])]
        # + ["cs_states{}".format(i) for i in range(cs_states.shape[1])]
    )
    df_output = pd.DataFrame(data=connected_data, columns=header)
    df_output.to_excel(RESULT_DIR + "output{:02}.xlsx".format(j + 1), index=False)

    # np.savetxt(
    #     RESULT_DIR + "{:02}.csv".format(j + 1), np_output, delimiter=",",
    # )
# alltype_pca = PCA(n_components=2).fit_transform(alltype_cs)
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
