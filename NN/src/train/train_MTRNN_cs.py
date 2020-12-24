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

    if argnum == 3:
        # _, name, in_size, outsize = sys.argv
        # name = str(name)
        # in_size = int(in_size)
        # out_size = int(outsize)
        _, cf_num, cs_num = sys.argv
        cf_num = int(cf_num)
        cs_num = int(cs_num)
    else:
        raise Exception("Fail arg num")
    # loss_rate = sys.argv[1:]
    # name = ""
    # for i in loss_rate:
    #     name += i + "_"
    # loss_rate = list(map(float, loss_rate))
    # print(loss_rate)

    in_size, out_size = 30, 30
    open_rate = 0.01

    load_path = ""  # input("?.pth:")
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    my_dir = "MTRNN/1223/cs/"
    DATA_DIR = CURRENT_DIR + "/../../data/" + my_dir
    TRAIN_PATH = DATA_DIR + "train"
    TEST_PATH = DATA_DIR + "test"
    MODEL_BASE = "/media/user/ボリューム/model/"
    MODEL_BASE = CURRENT_DIR + "/../../../../model/"
    MODEL_DIR = MODEL_BASE + my_dir + "{}_{}/".format(cf_num, cs_num)
    os.makedirs(MODEL_DIR)
    # MODEL_DIR = MODEL_BASE + "MTRNN/1116_noimg2/"

    trainset = MyDataSet(TRAIN_PATH)
    testset = []  # MyDataSet(TEST_PATH)
    # in_size = 30  # trainset[0][0].shape[1]
    net = MTRNN(
        len(trainset),
        layer_size={
            "in": in_size,
            "out": out_size,
            "io": 50,
            "cf": cf_num,  # 70,80,90,100
            "cs": cs_num,  # 8,10,12,15
        },
        tau={"tau_io": 2, "tau_cf": 10, "tau_cs": 30},
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
        # "dims": [7, 7, 16, 15],
        # "loss_rates": loss_rate,
        # "learn_rate": 0.01,
        # "betas": (0.999, 0.999),
    }
    criterion = torch.nn.MSELoss()

    class TrainNet(TrainBase):
        def __init__(
            self,
            net,
            criterion,
            trainset,
            testset,
            model_dir,
            param_dict={},
            load_path="",
            device=None,
        ):
            # default parameter
            self._param_dict = {
                "train_batch_size": 4,
                "test_batch_size": len(testset),
                "epoch": 30,
                "save_span": 50,
                "graph_span": 1,
                "weight_decay": 0,
                "learn_rate": 0.001,
                "betas": (0.9, 0.999),
            }

            self._param_dict.update(param_dict)
            self._net = net
            print(self._net)
            self._device = device
            if device is not None:
                self._net.to(device)
            print("num of (train,test)=({},{})".format(len(trainset), len(testset)))
            self._trainloader = torch.utils.data.DataLoader(
                trainset,
                batch_size=self._param_dict["train_batch_size"],
                shuffle=False,
                num_workers=os.cpu_count(),
            )
            self._testloader = []
            self._criterion = criterion
            self._optimizer = torch.optim.Adam(
                self._net.parameters(),
                lr=self._param_dict["learn_rate"],
                betas=self._param_dict["betas"],
                weight_decay=self._param_dict["weight_decay"],
            )
            self._model_dir = model_dir
            self._graph_epoch = []  # x dim of figure
            self._train_loss_value = []  # training loss list
            self._test_loss_value = []  # test loss list

            if load_path != "":
                self.load_model(load_path)

            torch.backends.cudnn.benchmark = True
            self.init()

        def _each_epoch(self, mode, dataloader):
            calc_num = 0
            sum_loss = 0
            for (one_batch_inputs, one_batch_labels) in dataloader:
                inputs_transposed = one_batch_inputs.transpose(1, 0)
                labels_transposed = one_batch_labels.transpose(1, 0)[:, :, :out_size]
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

            mean_loss = sum_loss / calc_num if calc_num > 0 else 0
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
