#!/usr/bin/env python3
# coding: utf-8
import os
from mydataset import EasyDataSet as MyDataSet
from MTRNN import CustomRNN, MTRNNCell

# from train_net_predict import TrainNet

from train_MTRNN import TrainNet
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = CURRENT_DIR + "/../data/"
# TRAIN_PATH = DATA_DIR + "train"
# TEST_PATH = DATA_DIR + "test"
MODEL_DIR = DATA_DIR + "model_MTRNN/"
one_sequence_size = 600  # traning dataのデータ数
trainset = MyDataSet(0, one_sequence_size, 0.02, 3)
testset = MyDataSet(1, one_sequence_size, 0.02, 1)

# # trainset = MyDataSet(training_size)
# testset = trainset
net = MTRNNCell()
# net = MTRNNCell()
## modelをロードしたとき
# model_path = MODEL_DIR + "20200612_103406_100.pth"
# net.load_state_dict(torch.load(model_path))
param_dict = {
    "train_batch_size": 1,
    "test_batch_size": 1,
    "epoch": 1,
    "save_span": 50,
    "graph_span": 5,
    "weight_decay": 0.00001,
    # "learn_rate": 0.12,
    # "betas": (0.999, 0.999),
}
criterion = torch.nn.MSELoss()
train_net = TrainNet(net, criterion, trainset, testset, MODEL_DIR, param_dict)
# train_net.load_model("20200626_225204_100finish")

train_net.run()
