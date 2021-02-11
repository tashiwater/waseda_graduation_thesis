#!/usr/bin/env python3
# coding: utf-8

import os
import sys
import torch
from torch.autograd import detect_anomaly

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from train_base import TrainBase
from dataset.dataset_CNNMTRNN import MyDataSet
from model.CNNMTRNN import CNNMTRNN as Net

if __name__ == "__main__":
    # argnum = len(sys.argv)
    # if argnum == 1:
    #     name = input("file name :")
    #     cf_num = int(input("cf_num (default = 200):"))
    #     cs_tau = int(input("cs_tau (default = 50):"))
    # elif argnum == 5:
    #     _, name, cf_num, cs_tau, open_rate = sys.argv
    #     cf_num = int(cf_num)
    #     cs_tau = int(cs_tau)
    #     open_rate = float(open_rate)
    # else:
    #     raise Exception("Fail arg num")
    cf_num = 100
    cs_tau = 50
    open_rate = 0.9

    load_path = input("?.pth:")
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = CURRENT_DIR + "/../../data/CNNMTRNN/"
    TRAIN_PATH = DATA_DIR + "motor/train"
    TEST_PATH = DATA_DIR + "motor/test"
    img_TRAIN_PATH = DATA_DIR + "img/train"
    img_TEST_PATH = DATA_DIR + "img/test"

    MODEL_BASE = "/media/user/ボリューム/model/"
    MODEL_BASE = CURRENT_DIR + "/../../../../model/"
    # MODEL_DIR = MODEL_BASE + "MTRNN/custom_loss/open_{:02}/{}/".format(
    #     int(open_rate * 10), name
    # )
    MODEL_DIR = MODEL_BASE + "CNNMTRNN/open1/"

    trainset = MyDataSet(
        TRAIN_PATH,
        img_TRAIN_PATH,
        img_size=(128, 96),
        is_test=False,
        dsize=5,
        noise=0.01,
    )
    testset = MyDataSet(
        TEST_PATH, img_TEST_PATH, img_size=(128, 96), is_test=True, dsize=5
    )
    in_size = 41
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
    param_dict = {
        "train_batch_size": len(trainset),
        "test_batch_size": len(testset),
        "epoch": None,
        "save_span": 50,
        "graph_span": 5,
        "weight_decay": 0.00001,
        # "learn_rate": 0.01,
        # "betas": (0.999, 0.999),
    }
    criterion = torch.nn.MSELoss()

    class TrainNet(TrainBase):
        def _each_epoch(self, mode, dataloader):
            calc_num = 0
            sum_loss = 0
            # with detect_anomaly():
            for (one_batch_inputs, one_batch_labels) in dataloader:
                input_motor, input_img = one_batch_inputs
                label_motor, label_img = one_batch_labels
                input_motor = input_motor.transpose(1, 0)
                input_img = input_img.transpose(1, 0)
                label_motor = label_motor.transpose(1, 0)
                label_img = label_img.transpose(1, 0)
                self._net.init_state(input_motor.shape[1])
                output_motor = torch.zeros_like(label_motor)
                output_img = torch.zeros_like(label_img)
                self._optimizer.zero_grad()

                for i, inputs_t in enumerate(zip(input_motor, input_img)):
                    inputs_t = (
                        inputs_t[0].to(self._device),
                        inputs_t[1].to(self._device),
                    )
                    output_motor[i], output_img[i] = self._net(inputs_t[0], inputs_t[1])

                    # if torch.isnan(output_img).any():
                    #     print("nan1")
                    # if torch.isnan(output_motor).any():
                #     print("nan1")
                loss = self._criterion(output_motor, label_motor) + self._criterion(
                    output_img, label_img
                )
                # loss = sum(loss)
                if mode == "train":
                    # sum(loss).backward()
                    # pritn(self._net.cs2cs.parameters.grad)
                    loss.backward()
                    self._optimizer.step()
                    # sum_loss += sum(loss).item()
                sum_loss += loss.item()
                calc_num += 1
            mean_loss = sum_loss / calc_num
            print(mode + " meanloss={}".format(mean_loss))
            return mean_loss

    train_net = TrainNet(
        net,
        criterion,
        trainset,
        testset,
        MODEL_DIR,
        param_dict,
        load_path,
        torch.device("cuda:0"),
    )
    train_net.run()
