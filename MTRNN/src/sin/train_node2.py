#!/usr/bin/env python3
# coding: utf-8
import os
from mydataset import EasyDataSet as MyDataSet
from MTRNN import MTRNN
from train import TrainNet
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = CURRENT_DIR + "/../data/"
TRAIN_PATH = DATA_DIR + "train"
TEST_PATH = DATA_DIR + "test"
MODEL_DIR = DATA_DIR + "model2/"
one_sequence_size = 600  # traning dataのデータ数
trainset = MyDataSet(0, one_sequence_size, 0.02, 300, 0.1)
testset = MyDataSet(1, one_sequence_size, 0.02, 1)

# trainset = MyDataSet(TRAIN_PATH)
# testset = MyDataSet(TEST_PATH)
# net = CustomRNN(MTRNNCell())
in_size = trainset[0][0].shape[1]
net = MTRNN(
    in_size=in_size,
    out_size=in_size,
    # c_size={"io": 34, "cf": 160, "cs": 13},
    c_size={"io": 4, "cf": 2, "cs": 2},
    tau={"tau_io": 2, "tau_cf": 5, "tau_cs": 50},
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
    "dims": [1, 1],
    "loss_rates": [0.5, 0.5],
    # "loss_rates": [0.25, 0.25, 0.25, 0.25],
    # "learn_rate": 0.12,
    # "betas": (0.999, 0.999),
}
criterion = torch.nn.MSELoss()
train_net = TrainNet(net, criterion, trainset, testset, MODEL_DIR, param_dict)
# train_net.load_model("20200708_175316_500")

train_net.run()
