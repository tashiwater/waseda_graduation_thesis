#!/usr/bin/env python3
# coding: utf-8

import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from train_base import TrainBase
from dataset.dataset_MTRNN import MyDataSet
from model.AttentionMTRNN import AttentionMTRNN as MTRNN

if __name__ == "__main__":
    # argnum = len(sys.argv)

    # if argnum == 4:
    #     _, name, in_size, outsize = sys.argv
    #     name = str(name)
    #     in_size = int(in_size)
    #     out_size = int(outsize)
    #     # _, cf_num, cs_num = sys.argv
    #     # cf_num = int(cf_num)
    #     # cs_num = int(cs_num)
    # else:
    #     raise Exception("Fail arg num")
    open_rate = 0.1
    cf_num, cs_num = 80, 10
    load_path = ""  # input("?.pth:")
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = CURRENT_DIR + "/../../data/MTRNN_all/"
    TRAIN_PATH = DATA_DIR + "train"
    TEST_PATH = DATA_DIR + "test"
    MODEL_BASE = "/media/user/ボリューム/model/"
    MODEL_BASE = CURRENT_DIR + "/../../../../model/"
    MODEL_DIR = MODEL_BASE + "MTRNN/1123/attention_after_relu/"
    # os.makedirs(MODEL_DIR)
    # MODEL_DIR = MODEL_BASE + "MTRNN/1116_noimg2/"

    trainset = MyDataSet(TRAIN_PATH)
    testset = MyDataSet(TEST_PATH)
    in_size = 45  # trainset[0][0].shape[1]
    net = MTRNN(
        layer_size={
            "in": in_size,
            "out": in_size,
            "io": 50,
            "cf": cf_num,  # 70,80,90,100
            "cs": cs_num,  # 8,10,12,15
        },
        tau={"tau_io": 2, "tau_cf": 5, "tau_cs": 30},
        open_rate=open_rate,
        activate=torch.nn.Tanh(),
    )
    param_dict = {
        "train_batch_size": 1,
        "test_batch_size": 1,
        "epoch": 10000,
        "save_span": 100,
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
            for (one_batch_inputs, one_batch_labels) in dataloader:
                inputs_transposed = one_batch_inputs.transpose(1, 0)
                labels_transposed = one_batch_labels.transpose(1, 0)
                self._net.init_state(inputs_transposed.shape[1])
                outputs = torch.zeros_like(labels_transposed)
                # label_attention = torch.zeros_like(labels_transposed)
                self._optimizer.zero_grad()
                torch.autograd.set_detect_anomaly(True)
                loss = []
                for i, inputs_t in enumerate(inputs_transposed):
                    inputs_t = inputs_t.to(self._device)
                    outputs[i] = self._net(inputs_t)
                    attention_map = self._net.attention_map.detach()
                    loss += [
                        self._criterion(outputs[i, :, :7], labels_transposed[i, :, :7]),
                        self._criterion(
                            outputs[i, :, 7:30], labels_transposed[i, :, 7:30]
                        )
                        * attention_map[0, 7],
                        self._criterion(
                            outputs[i, :, 30:], labels_transposed[i, :, 30:]
                        )
                        * attention_map[0, 30],
                    ]

                loss = sum(loss)
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
        # torch.device("cuda:0"),
    )
    train_net.run()
