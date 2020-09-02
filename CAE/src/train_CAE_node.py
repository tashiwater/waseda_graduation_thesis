#!/usr/bin/env python3
# coding: utf-8
import os
from mydataset import MyDataSetForCAE as MyDataSet
from CAE import CAE as Net
from train import TrainNet
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = CURRENT_DIR + "/../data/"
TRAIN_PATH = DATA_DIR + "train"
TEST_PATH = DATA_DIR + "test"
MODEL_DIR = DATA_DIR + "model_CAE20/sigmoid/"
trainset = MyDataSet(TRAIN_PATH, noise=0.01)
testset = MyDataSet(TEST_PATH)
net = Net()
param_dict = {
    "train_batch_size": 500,
    "test_batch_size": 500,
    "epoch": None,
    "save_span": 50,
    "graph_span": 5,
    "weight_decay": 0.00001,
    "learn_rate": 0.00001,
    # "betas": (0.999, 0.999),
}
criterion = torch.nn.MSELoss()
train_net = TrainNet(net, criterion, trainset, testset, MODEL_DIR, param_dict)
# train_net.load_model("20200831_120247_250")
train_net.run()
