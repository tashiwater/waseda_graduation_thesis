#!/usr/bin/env python3
# coding: utf-8

import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from train_base import TrainBase
from dataset.dataset_CAE import MyDataSet
from model.CAE import CAE as Model

if __name__ == "__main__":
    load_path = ""  # input("?.pth:")
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = CURRENT_DIR + "/../../data/CAE/"
    TRAIN_PATH = DATA_DIR + "train"
    TEST_PATH = DATA_DIR + "test"
    MODEL_BASE = "/media/user/ボリューム/model/"
    MODEL_BASE = CURRENT_DIR + "/../../../../model/"
    MODEL_DIR = MODEL_BASE + "CAE/1123/"

    trainset = MyDataSet(
        TRAIN_PATH, img_size=(128, 96), is_test=False, dsize=5, noise=0.01
    )
    testset = MyDataSet(TEST_PATH, img_size=(128, 96), is_test=True, dsize=5)
    net = Model()
    param_dict = {
        "train_batch_size": 500,
        "test_batch_size": 500,
        "epoch": 3000,
        "save_span": 50,
        "graph_span": 5,
        "weight_decay": 0.00001,
    }
    criterion = torch.nn.MSELoss()

    class TrainNet(TrainBase):
        def _each_epoch(self, mode, dataloader):
            calc_num = 0
            sum_loss = 0
            for (inputs, labels) in dataloader:
                if self._device is not None:
                    inputs, labels = (
                        inputs.to(self._device),
                        [labels[i].to(self._device) for i in range(2)],
                    )
                self._optimizer.zero_grad()
                outputs = self._net(inputs)
                # loss = [
                #     self._criterion[i](outputs[i], labels[i]) * self._criterion_rate[i]
                #     for i in range(2)
                # ]
                loss = self._criterion(outputs, labels[0])
                # loss = sum(loss)
                if mode == "train":
                    loss.backward()
                    self._optimizer.step()
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
