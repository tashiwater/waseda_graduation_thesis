#!/usr/bin/env python3
# coding: utf-8

import os
import sys
import torch
import torchvision
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dataset.dataset_CNNMTRNN import MyDataSet
from model.CNNMTRNN import CNNMTRNN as Net

if __name__ == "__main__":
    cf_num = 100
    cs_tau = 50
    open_rate = 0.1

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = CURRENT_DIR + "/../../data/CNNMTRNN/"
    TRAIN_PATH = DATA_DIR + "motor/train"
    TEST_PATH = DATA_DIR + "motor/test"
    img_TRAIN_PATH = DATA_DIR + "img/train"
    img_TEST_PATH = DATA_DIR + "img/test"
    RESULT_DIR = DATA_DIR + "result/motor/"
    img_RESULT_DIR = DATA_DIR + "result/img/"
    MODEL_BASE = "/media/user/ボリューム/model/"
    MODEL_BASE = CURRENT_DIR + "/../../../../model/"
    # MODEL_DIR = MODEL_BASE + "MTRNN/custom_loss/open_{:02}/{}/".format(
    #     int(open_rate * 10), name
    # )
    MODEL_DIR = MODEL_BASE + "CNNMTRNN/"
    load_path = "open1/20201110_042645_1500"  # input("?aa.pth:")

    dataset = MyDataSet(
        TEST_PATH, img_TEST_PATH, img_size=(128, 96), is_test=True, dsize=5
    )
    in_size = 41  # trainset[0][0].shape[1]
    position_dims = 7
    net = Net(
        layer_size={
            "in": in_size,
            "out": in_size,
            "io": 50,
            "cf": cf_num,
            "cs": 15,
        },
        tau={"tau_io": 2, "tau_cf": 5, "tau_cs": cs_tau},
        open_rate=open_rate,
    )
    model_path = MODEL_DIR + load_path + ".pth"
    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint["model"])

    device = torch.device("cuda:0")
    criterion = torch.nn.MSELoss()
    net = net.to(device)

    def get_header(add_word):
        return (
            [add_word + "position{}".format(i) for i in range(7)]
            + [add_word + "torque{}".format(i) for i in range(7)]
            + [add_word + "tactile{}".format(i) for i in range(12)]
        )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
    )

    net.eval()
    count = 0
    alltype_cs = []
    for j, (one_batch_inputs, one_batch_labels) in enumerate(dataloader):
        input_motor, input_img = one_batch_inputs
        label_motor, label_img = one_batch_labels
        input_motor = input_motor.transpose(1, 0)
        input_img = input_img.transpose(1, 0)
        label_motor = label_motor.transpose(1, 0)
        label_img = label_img.transpose(1, 0)
        net.init_state(input_motor.shape[1])
        output_motor = torch.zeros_like(label_motor)
        output_img = torch.zeros_like(label_img)

        for i, inputs_t in enumerate(zip(input_motor, input_img)):
            inputs_t = (
                inputs_t[0].to(device),
                inputs_t[1].to(device),
            )
            output_motor[i], output_img[i] = net(inputs_t[0], inputs_t[1])

        posi_loss = criterion(output_motor, label_motor)
        img_loss = criterion(output_img, label_img)
        print("loss={} / {}".format(posi_loss.item(), img_loss.item()))

        np_input = label_motor.view(-1, 26).detach().numpy()
        np_output = output_motor.view(-1, 26).detach().numpy()
        # cs_pca = PCA(n_components=2).fit_transform(cs_states)
        # cf_pca = PCA(n_components=2).fit_transform(cf_states)
        connected_data = np.hstack(
            [
                np_input,
                np_output,
                # cf_pca,
                # cs_pca,
                # cs_states
                # attention_map,
            ]  # , io_states, cf_states, cs_states
        )
        header = (
            # ["input{}".format(i) for i in range(np_input.shape[1])]
            # + ["output{}".format(i) for i in range(np_output.shape[1])]
            get_header("in ")
            + get_header("out ")
            # + ["cf_pca{}".format(i) for i in range(cf_pca.shape[1])]
            # + ["cs_pca{}".format(i) for i in range(cs_pca.shape[1])]
            # # + ["attention_map{}".format(i) for i in range(attention_map.shape[1])]
            # # + ["io_states{}".format(i) for i in range(io_states.shape[1])]
            # # + ["cf_states{}".format(i) for i in range(cf_states.shape[1])]
            # + ["cs_states{}".format(i) for i in range(cs_states.shape[1])]
        )
        df_output = pd.DataFrame(data=connected_data, columns=header)
        df_output.to_excel(RESULT_DIR + "output{:02}.xlsx".format(j + 1), index=False)

        for img in output_img.cpu():
            torchvision.utils.save_image(
                img, img_RESULT_DIR + "{:04d}.png".format(count)
            )
            count += 1
        df_output.iloc[:, :7].plot()
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.title("input")
        plt.subplots_adjust(right=0.7)
        # plt.show()
        df_output.iloc[:, 26:33].plot()
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.title("output")
        plt.subplots_adjust(right=0.7)
        plt.show()
