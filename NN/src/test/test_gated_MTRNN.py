#!/usr/bin/env python3
# coding: utf-8

import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from train_base import TrainBase
from dataset.dataset_gated_MTRNN import MyDataSet
from model.GatedMTRNN import GatedMTRNN

if __name__ == "__main__":
    cf_num = 100
    cs_tau = 50
    open_rate = 0.1

    load_path = ""  # input("?aa.pth:")
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = CURRENT_DIR + "/../../data/GatedMTRNN/"
    TEST_PATH = DATA_DIR + "test"
    MODEL_BASE = "/media/hdd_1tb/model/"
    # MODEL_BASE = CURRENT_DIR + "/../../../../model/"
    # MODEL_DIR = MODEL_BASE + "MTRNN/custom_loss/open_{:02}/{}/".format(
    #     int(open_rate * 10), name
    # )
    MODEL_DIR = MODEL_BASE + "GatedMTRNN/default/"

    testset = MyDataSet(TEST_PATH)
    in_size = 41  # trainset[0][0].shape[1]
    position_dims = 7
    net = GatedMTRNN(
        layer_size={
            "in": in_size,
            "out": position_dims,
            "io": 50,
            "cf": cf_num,
            "cs": 15,
        },
        tau={"tau_io": 2, "tau_cf": 5, "tau_cs": cs_tau},
        open_rate=open_rate,
    )
    criterion = torch.nn.MSELoss()

    def get_header(add_word):
        return (
            [add_word + "position{}".format(i) for i in range(7)]
            + [add_word + "torque{}".format(i) for i in range(7)]
            + [add_word + "tactile{}".format(i) for i in range(12)]
            + [add_word + "image{}".format(i) for i in range(20)]
        )

    dataloader = torch.utils.data.DataLoader(
        testset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
    )

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
            # pri(outputs[i])
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
