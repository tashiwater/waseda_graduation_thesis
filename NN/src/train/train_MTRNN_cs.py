#!/usr/bin/env python3
# coding: utf-8

import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from train_base import TrainBase
from dataset.dataset_MTRNN import MyDataSet
from model.MTRNN_cs import MTRNN

if __name__ == "__main__":
    argnum = len(sys.argv)

    if argnum == 5:
        # _, name, in_size, outsize = sys.argv
        # name = str(name)
        # in_size = int(in_size)
        # out_size = int(outsize)
        _, cf_num, cs_num, cf_tau, cs_tau = sys.argv
        cf_num = int(cf_num)
        cs_num = int(cs_num)
        cf_tau = int(cf_tau)
        cs_tau = int(cs_tau)

    open_rate = 0.01
    in_size, out_size = 45, 45
    load_path = ""  # input("?.pth:")
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    my_dir = "MTRNN/0106/all_cs0/"
    DATA_DIR = CURRENT_DIR + "/../../data/" + my_dir
    TRAIN_PATH = DATA_DIR + "train"
    TEST_PATH = DATA_DIR + "test"
    MODEL_BASE = "/media/user/ボリューム/model/"
    MODEL_BASE = CURRENT_DIR + "/../../../../model/"
    MODEL_DIR = (
        MODEL_BASE + my_dir + "cf{}cs{}/{}_{}/".format(cf_tau, cs_tau, cf_num, cs_num)
    )
    os.makedirs(MODEL_DIR)
    # MODEL_DIR = MODEL_BASE + "MTRNN/1116_noimg2/"

    trainset = MyDataSet(TRAIN_PATH)
    testset = MyDataSet(TEST_PATH)
    # in_size = 30  # trainset[0][0].shape[1]
    position_dims = 7
    net = MTRNN(
        len(trainset),
        layer_size={
            "in": in_size,
            "out": out_size,
            "io": 50,
            "cf": cf_num,  # 70,80,90,100
            "cs": cs_num,  # 8,10,12,15
        },
        tau={"tau_io": 2, "tau_cf": cf_tau, "tau_cs": cs_tau},
        open_rate=open_rate,
        activate=torch.nn.Tanh(),
    )
    param_dict = {
        "train_batch_size": len(trainset),
        "test_batch_size": len(testset),
        "epoch": 5000,
        "save_span": 100,
        "graph_span": 5,
        "weight_decay": 0.00001,
        "dims": [45],
        "loss_rates": [1],
        # "learn_rate": 0.01,
        # "betas": (0.999, 0.999),
    }
    criterion = torch.nn.MSELoss()

    class TrainNet(TrainBase):
        def set_data_loader(self, trainset, testset):
            self._trainloader = torch.utils.data.DataLoader(
                trainset,
                batch_size=self._param_dict["train_batch_size"],
                shuffle=False,
                num_workers=os.cpu_count(),
            )
            self._testloader = torch.utils.data.DataLoader(
                testset,
                batch_size=self._param_dict["test_batch_size"],
                shuffle=False,
                num_workers=os.cpu_count(),
            )

        def _each_epoch(self, mode, dataloader):
            calc_num = 0
            sum_loss = 0
            for j, (one_batch_inputs, one_batch_labels) in enumerate(dataloader):
                inputs_transposed = one_batch_inputs.transpose(1, 0)
                labels_transposed = one_batch_labels.transpose(1, 0)[:, :, :out_size]
                if mode == "test":
                    each_container = 3
                    cs0 = self._net.cs0[::each_container]
                    self._net.init_state(inputs_transposed.shape[1], cs0)
                else:
                    self._net.init_state(inputs_transposed.shape[1])
                outputs = torch.zeros_like(labels_transposed)
                self._optimizer.zero_grad()
                for i, inputs_t in enumerate(inputs_transposed):
                    inputs_t = inputs_t.to(self._device)
                    outputs[i] = self._net(inputs_t)
                    """
                d1 = 0
                loss = []
                for dim, rate in zip(
                    self._param_dict["dims"], self._param_dict["loss_rates"]
                ):
                    d2 = d1 + dim
                    one_loss = self._criterion(
                        outputs[:, :, d1:d2], labels_transposed[:, :, d1:d2]
                    )
                    loss.append(one_loss * rate)
                    d1 = d2
                    """
                loss = self._criterion(outputs, labels_transposed)
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
        # torch.device("cuda:0"),
    )
    train_net.run()
