#!/usr/bin/env python3
# coding: utf-8
import os
from mydataset import MyDataSetForAttention as MyDataSet
from CAE import CAEwithAttention as Net
from train_attention import TrainNet
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = CURRENT_DIR + "/../data/"
TRAIN_PATH = DATA_DIR + "train_attention"
TEST_PATH = DATA_DIR + "test_attention"
MODEL_DIR = DATA_DIR + "model_CAE20/attention/"
trainset = MyDataSet(TRAIN_PATH, noise=0.01)
testset = MyDataSet(TEST_PATH)
net = Net()
param_dict = {
    "train_batch_size": 500,
    "test_batch_size": 500,
    "epoch": None,
    "save_span": 50,
    "graph_span": 5,
    # "weight_decay": 0.00001,
    # "learn_rate": 0.00001,
    # "betas": (0.999, 0.999),
}
# criterion = torch.nn.MSELoss()
criterion = [torch.nn.MSELoss(), torch.nn.CrossEntropyLoss()]
criterion_rate = [1, 0.001]
train_net = TrainNet(
    net,
    criterion,
    trainset,
    testset,
    MODEL_DIR,
    param_dict,
    device=torch.device("cuda:0"),
    criterion_rate=criterion_rate,
)
# train_net.load_model("20200911_174618_200")
train_net.run()
