#!/usr/bin/env python3
# coding: utf-8

import os
import sys
import torch
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dataset.dataset_MTRNN import MyDataSet
from model.Takahashi2 import AttentionMTRNN as Net

if __name__ == "__main__":
    is_print = True
    cf_num = 80
    cs_num = 10
    open_rate = 1

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = CURRENT_DIR + "/../../data/MTRNN_all/"
    TEST_PATH = DATA_DIR + "test"
    RESULT_DIR = DATA_DIR + "result/"
    MODEL_BASE = "/media/user/ボリューム/model/"
    MODEL_BASE = CURRENT_DIR + "/../../../../model/"
    # MODEL_BASE = DATA_DIR + "../model/"
    # MODEL_DIR = MODEL_BASE + "MTRNN/custom_loss/open_{:02}/{}/".format(
    #     int(open_rate * 10), name
    # )
    MODEL_DIR = MODEL_BASE + "MTRNN/"
    load_path = "1123/attentions/takahashiplus_sigmoid_noloss/20201126_142614_5000finish"  # input("?aa.pth:")
    # load_path = "1119_70_8/20201120_001102_10000finish"
    dataset = MyDataSet(TEST_PATH)
    in_size = 45  # trainset[0][0].shape[1]
    net = Net(
        layer_size={
            "in": in_size,
            "out": in_size,
            "io": 50,
            "cf": cf_num,
            "cs": cs_num,
        },
        tau={"tau_io": 2, "tau_cf": 5, "tau_cs": 30},
        open_rate=open_rate,
        # activate=torch.nn.Softmax(dim=1),
        activate=torch.nn.Sigmoid(),
    )
    model_path = MODEL_DIR + load_path + ".pth"
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    net.load_state_dict(checkpoint["model"])

    criterion = torch.nn.MSELoss()

    def get_header(add_word):
        return (
            [add_word + "position{}".format(i) for i in range(7)]
            + [add_word + "torque{}".format(i) for i in range(7)]
            + [add_word + "tactile{}".format(i) for i in range(16)]
            + [add_word + "image{}".format(i) for i in range(15)]
        )

    dataloader = torch.utils.data.DataLoader(
        dataset,
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
        attention_map = torch.zeros_like(labels_transposed)
        extracted = torch.zeros_like(labels_transposed)
        cs_states = []
        for i, inputs_t in enumerate(inputs_transposed):
            outputs[i] = net(inputs_t)
            attention_map[i] = net.attention_map
            extracted[i] = net.extracted
            cs_states.append(net.mtrnn.cs_state.view(-1).detach().numpy())
            # attention_map.append(net.attention_map.view(-1).detach().numpy())
        posi_loss = criterion(outputs[:, :, :7], labels_transposed[:, :, :7])
        tactile_loss = criterion(outputs[:, :, 7:30], labels_transposed[:, :, 7:30])
        img_loss = criterion(outputs[:, :, 30:], labels_transposed[:, :, 30:])
        loss = criterion(outputs, labels_transposed)
        print(
            "loss={}, {}, {} / {}".format(
                posi_loss.item(), tactile_loss.item(), img_loss.item(), loss.item()
            )
        )

        cs_states = np.array(cs_states)
        np_input = labels_transposed.view(-1, in_size).detach().numpy()
        np_output = outputs.view(-1, in_size).detach().numpy()
        attention_map = attention_map.view(-1, in_size).detach().numpy()
        extracted = extracted.view(-1, in_size).detach().numpy()
        connected_data = np.hstack(
            [
                np_input,
                np_output,
                attention_map,
                cs_states,
                extracted
                # attention_map,
            ]  # , io_states, cf_states, cs_states
        )
        header = (
            # ["input{}".format(i) for i in range(np_input.shape[1])]
            # + ["output{}".format(i) for i in range(np_output.shape[1])]
            get_header("in ")
            + get_header("out ")
            + get_header("attention ")
            # + ["attention_map{}".format(i) for i in range(attention_map.shape[1])]
            # + ["io_states{}".format(i) for i in range(io_states.shape[1])]
            # + ["cf_states{}".format(i) for i in range(cf_states.shape[1])]
            + ["cs_states{}".format(i) for i in range(cs_states.shape[1])]
            + get_header("extracted ")
        )
        df_output = pd.DataFrame(data=connected_data, columns=header)
        df_output.to_excel(RESULT_DIR + "output{:02}.xlsx".format(j + 1), index=False)

        if is_print:
            df_output.iloc[:, :7].plot()
            plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
            plt.title("input")
            plt.subplots_adjust(right=0.7)
            # plt.show()
            df_output.iloc[:, 45:52].plot()
            plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
            plt.title("output")
            plt.subplots_adjust(right=0.7)

            df_output.iloc[:, 119:121].plot()
            plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
            plt.title("attention")
            plt.subplots_adjust(right=0.7)
            plt.show()

            df_output.iloc[:, 152:175].plot()
            plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
            plt.title("extracted")
            plt.subplots_adjust(right=0.7)
            plt.show()
