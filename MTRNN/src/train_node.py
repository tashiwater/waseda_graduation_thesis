#!/usr/bin/env python3
# coding: utf-8
import os

import torch

### normal MTRNN
# from mydataset import CsvDataSet as MyDataSet
# from MTRNN import MTRNN
# from train import TrainNet
### custom MTRNN
from MTRNN import CustomNet as MTRNN
from mydataset import CustomDataSet as MyDataSet
from train_custom import TrainNet

name = input("file name :")
cf_num = int(input("cf_num (default = 200):"))
cs_tau = int(input("cs_tau (default = 50):"))
load_path = input("?.pth:")
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = CURRENT_DIR + "/../data/"
TRAIN_PATH = DATA_DIR + "train"
TEST_PATH = DATA_DIR + "test"
MODEL_DIR = CURRENT_DIR + "/../../../model/customMTRNN/{}/".format(name)
# MODEL_DIR = "/media/hdd_1tb/model/MTRNN/{}/".format(name)
# one_sequence_size = 600  # traning dataのデータ数
# trainset = MyDataSet(0, one_sequence_size, 0.02, 3, 0.1)
# testset = MyDataSet(1, one_sequence_size, 0.02, 1)

trainset = MyDataSet(TRAIN_PATH)
testset = MyDataSet(TEST_PATH)
in_size = 46  # trainset[0][0].shape[1]
net = MTRNN(
    layer_size={"in": in_size, "out": in_size, "io": 34, "cf": cf_num, "cs": 15},
    tau={"tau_io": 2, "tau_cf": 5, "tau_cs": cs_tau},
    open_rate=0.1,
)
## modelをロードしたとき
# model_path = MODEL_DIR + "20200612_103406_100.pth"
# net.load_state_dict(torch.load(model_path))
param_dict = {
    "train_batch_size": len(trainset),
    "test_batch_size": len(testset),
    "epoch": None,
    "save_span": 500,
    "graph_span": 5,
    "weight_decay": 0.00001,
    "dims": [in_size],
    "loss_rates": [1],
    # "learn_rate": 0.01,
    # "betas": (0.999, 0.999),
}
criterion = torch.nn.MSELoss()
train_net = TrainNet(
    net,
    criterion,
    trainset,
    testset,
    MODEL_DIR,
    param_dict,
)
if load_path != "":
    train_net.load_model(load_path)
train_net.run()
