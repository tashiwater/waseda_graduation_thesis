#!/usr/bin/env python3
# coding: utf-8
import os
from mydataset import CustomDataSet as MyDataSet
from MTRNN import CustomNet as MTRNN
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
MODEL_BASE = "/media/hdd_1tb/model/"
# MODEL_BASE =  CURRENT_DIR + "/../../../model/"
MODEL_DIR = MODEL_BASE + "MTRNN/custom/"
VALIDATE_PATH = DATA_DIR + "test"

dataset = MyDataSet(VALIDATE_PATH)
# dataset = MyDataSet(0, 600, 0.02, 1, 0.1)
in_size = 46
net = MTRNN(
    layer_size={"in": in_size, "out": in_size, "io": 50, "cf": 350, "cs": 15},
    tau={"tau_io": 2, "tau_cf": 5, "tau_cs": 50},
    open_rate=0.1,
)
### modelをロード
model_path = MODEL_DIR + "open_01/cf350/20201015_151528_18500.pth"
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
    motor, tactile, img = one_batch_inputs
    motor = motor.transpose(1, 0)
    tactile = tactile.transpose(1, 0)
    img = img.transpose(1, 0)
    labels_transposed = one_batch_labels.transpose(1, 0)
    net.init_state(labels_transposed.shape[1])
    outputs = torch.zeros_like(labels_transposed)
    io_states = []
    cf_states = []
    cs_states = []
    for i, inputs_t in enumerate(zip(motor, tactile, img)):
        outputs[i] = net(inputs_t[0], inputs_t[1], inputs_t[2])
        # pri(outputs[i])
        io_states.append(net.mtrnn.io_state.view(-1).detach().numpy())
        cf_states.append(net.mtrnn.cf_state.view(-1).detach().numpy())
        cs_states.append(net.mtrnn.cs_state.view(-1).detach().numpy())
    loss = criterion(outputs, labels_transposed)
    print("loss={}".format(loss.item()))
    loss2 = criterion(outputs[:, :, :7], labels_transposed[:, :, :7])
    print("position loss={}".format(loss2.item()))
    alltype_cs += cs_states
    io_states = np.array(io_states)
    cf_states = np.array(cf_states)
    cs_states = np.array(cs_states)

    np_input = labels_transposed.view(-1, in_size).detach().numpy()
    np_output = outputs.view(-1, in_size).detach().numpy()
    cs_pca = PCA(n_components=2).fit_transform(cs_states)
    cf_pca = PCA(n_components=2).fit_transform(cf_states)
    connected_data = np.hstack(
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

    df_in_posi = pd.DataFrame(np_input[:, :7])
    df_out_posi = pd.DataFrame(np_output[:, :7])
    df_in_posi.plot()
    df_out_posi.plot()
    plt.show()
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
