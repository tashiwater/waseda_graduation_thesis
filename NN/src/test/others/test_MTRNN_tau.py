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
from model.MTRNN_tau import MTRNN as Net

if __name__ == "__main__":
    is_print = True
    cf_num = 80
    cs_num = 15
    open_rate = 1

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = CURRENT_DIR + "/../../data/MTRNN_noimg/"
    TEST_PATH = DATA_DIR + "test"
    RESULT_DIR = DATA_DIR + "result/"
    MODEL_BASE = "/media/user/ボリューム/model/"
    MODEL_BASE = CURRENT_DIR + "/../../../../model/"
    # MODEL_BASE = DATA_DIR + "../model/"
    # MODEL_DIR = MODEL_BASE + "MTRNN/custom_loss/open_{:02}/{}/".format(
    #     int(open_rate * 10), name
    # )
    MODEL_DIR = MODEL_BASE + "MTRNN/"
    load_path = "1203/tau/5000/{}_{}".format(cf_num, cs_num)
    # load_path = "1119_70_8/20201120_001102_10000finish"
    dataset = MyDataSet(TEST_PATH)
    in_size = 30  # trainset[0][0].shape[1]
    position_dims = 7
    net = Net(
        24,
        layer_size={
            "in": in_size,
            "out": in_size,
            "io": 50,
            "cf": cf_num,
            "cs": cs_num,
        },
        tau={"tau_io": 2, "tau_cf": 5, "tau_cs": 30},
        open_rate=open_rate,
        activate=torch.nn.Tanh(),
    )
    model_path = MODEL_DIR + load_path + ".pth"
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    net.load_state_dict(checkpoint["model"])

    cs0 = net.cs0.detach().numpy().copy()
    if is_print:
        components = 2
        pca_base = PCA(n_components=components)
        pca_cs = pca_base.fit_transform(cs0)

        container_num = 6
        each_container = 3
        theta_num = 2

        stack = [
            pca_cs[i * each_container : (i + 1) * each_container] for i in range(6 * 2)
        ]

        colorlist = ["r", "g", "b", "c", "m", "y", "k"]
        # for i in range(185):
        fig = plt.figure()
        for i in range(components):
            axis1 = i
            for j in range(components - i - 1):
                axis2 = 1 + j + i
                for k in range(container_num):
                    a = k * 2
                    plt.scatter(
                        stack[a][:, axis1],
                        stack[a][:, axis2],
                        label="{} theta0".format(k),
                        edgecolors=colorlist[k],
                        facecolor="None",
                        marker="o",
                    )

                    n = 2 * k + 1  # container_num
                    plt.scatter(
                        stack[n][:, axis1],
                        stack[n][:, axis2],
                        label="{} theta30".format(k),
                        edgecolors=colorlist[k],
                        facecolor="None",
                        marker="D",
                    )

                plt.xlabel("pca{}".format(axis1 + 1))
                plt.ylabel("pca{}".format(axis2 + 1))
                plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
                plt.title("cs0")
                plt.subplots_adjust(right=0.7)
                plt.show()

    # print(cs0)
    np.savetxt(RESULT_DIR + "out.csv", cs0, delimiter=",")

    criterion = torch.nn.MSELoss()

    def get_header(add_word):
        return (
            [add_word + "position{}".format(i) for i in range(7)]
            + [add_word + "torque{}".format(i) for i in range(7)]
            + [add_word + "tactile{}".format(i) for i in range(16)]
            # + [add_word + "image{}".format(i) for i in range(15)]
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
        # if j < 11:
        #     continue
        inputs_transposed = one_batch_inputs.transpose(1, 0)
        labels_transposed = one_batch_labels.transpose(1, 0)
        net.init_state(inputs_transposed.shape[1], net.cs0[j * 3].unsqueeze(0))
        outputs = torch.zeros_like(labels_transposed)
        io_states = []
        cf_states = []
        cs_states = []
        attention_map = []
        tau = []
        for i, inputs_t in enumerate(inputs_transposed):
            outputs[i] = net(inputs_t)
            # for d in net.attention.parameters():
            #     print(d)
            # input()
            io_states.append(net.io_state.view(-1).detach().numpy())
            cf_states.append(net.cf_state.view(-1).detach().numpy())
            cs_states.append(net.cs_state.view(-1).detach().numpy())
            tau.append(net.cf_tau.view(-1).detach().numpy())
            # attention_map.append(net.attention_map.view(-1).detach().numpy())
        posi_loss = criterion(outputs[:, :, :7], labels_transposed[:, :, :7])
        loss = criterion(outputs, labels_transposed)
        print("loss={} / {}".format(posi_loss.item(), loss.item()))

        alltype_cs += cs_states
        io_states = np.array(io_states)
        cf_states = np.array(cf_states)
        cs_states = np.array(cs_states)
        tau = np.array(tau)
        # attention_map = np.array(attention_map)

        np_input = labels_transposed.view(-1, in_size).detach().numpy()
        np_output = outputs.view(-1, in_size).detach().numpy()
        cs_pca = PCA(n_components=2).fit_transform(cs_states)
        cf_pca = PCA(n_components=2).fit_transform(cf_states)
        connected_data = np.hstack(
            [
                np_input,
                np_output,
                cf_pca,
                cs_pca,
                cs_states,
                tau
                # attention_map,
            ]  # , io_states, cf_states, cs_states
        )
        header = (
            # ["input{}".format(i) for i in range(np_input.shape[1])]
            # + ["output{}".format(i) for i in range(np_output.shape[1])]
            get_header("in ")
            + get_header("out ")
            + ["cf_pca{}".format(i) for i in range(cf_pca.shape[1])]
            + ["cs_pca{}".format(i) for i in range(cs_pca.shape[1])]
            # + ["attention_map{}".format(i) for i in range(attention_map.shape[1])]
            # + ["io_states{}".format(i) for i in range(io_states.shape[1])]
            # + ["cf_states{}".format(i) for i in range(cf_states.shape[1])]
            + ["cs_states{}".format(i) for i in range(cs_states.shape[1])]
            + ["tau{}".format(i) for i in range(tau.shape[1])]
        )
        df_output = pd.DataFrame(data=connected_data, columns=header)
        df_output.to_excel(RESULT_DIR + "output{:02}.xlsx".format(j + 1), index=False)

        if is_print:
            ax = df_output.iloc[:, :7].plot(colormap="Accent", linestyle="--")
            # plt.plot(df_output.iloc[:, :7])
            # plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
            # plt.title("input")
            # plt.subplots_adjust(right=0.7)
            # plt.show()
            df_output.iloc[:, in_size : in_size + 7].plot(colormap="Accent", ax=ax)
            # plt.plot(df_output.iloc[:, in_size : in_size + 7], linestyle="-")
            plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
            # plt.title("output")
            plt.subplots_adjust(right=0.7)
            plt.show()
