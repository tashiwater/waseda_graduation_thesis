#!/usr/bin/env python3
# coding: utf-8
import os
from mydataset import EasyDataSet as MyDataSet
from MTRNN import CustomRNN, MTRNNCell

# from train_net_predict import TrainNet

from train_MTRNN2 import TrainNet
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = CURRENT_DIR + "/../data/"
TRAIN_PATH = DATA_DIR + "train"
TEST_PATH = DATA_DIR + "test"
MODEL_DIR = DATA_DIR + "model_MTRNN2/"
one_sequence_size = 700  # traning dataのデータ数
trainset = MyDataSet(0, one_sequence_size, 0.02, 3, 0.1)
testset = MyDataSet(1, one_sequence_size, 0.02, 1)

# trainset = MyDataSet(TRAIN_PATH)
# testset = MyDataSet(TEST_PATH)
# net = CustomRNN(MTRNNCell())
# print(trainset.size)
net = MTRNNCell(
    input_output_size=2,
    io_hidden_size=34,
    fast_hidden_size=160,
    slow_hidden_size=13,
    tau_input_output=2,
    tau_fast_hidden=5,
    tau_slow_hidden=50,
)
## modelをロードしたとき
# model_path = MODEL_DIR + "20200612_103406_100.pth"
# net.load_state_dict(torch.load(model_path))
param_dict = {
    "train_batch_size": 1,
    "test_batch_size": 1,
    "epoch": None,
    "save_span": 50,
    "graph_span": 5,
    "weight_decay": 0.00001,
    # "learn_rate": 0.12,
    # "betas": (0.999, 0.999),
}
criterion = torch.nn.MSELoss()
train_net = TrainNet(net, criterion, trainset, testset, MODEL_DIR, param_dict)
train_net.load_model("20200705_115808_1550")

train_net.run()
