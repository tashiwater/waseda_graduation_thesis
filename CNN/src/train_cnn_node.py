#!/usr/bin/env python3
# coding: utf-8
import os
from mydataset import MyDataSetForCNN as MyDataSet
from cnn import Net
from train_net import TrainNet
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = CURRENT_DIR + "/../data/"
TRAIN_PATH = DATA_DIR + "train"
TEST_PATH = DATA_DIR + "test"
MODEL_DIR = DATA_DIR + "model_cnn/"
trainset = MyDataSet(TRAIN_PATH)
testset = MyDataSet(TEST_PATH)

net = Net()

### modelをロードしたとき
# model_path = MODEL_DIR + "20200611_162949_20.pth"
# net.load_state_dict(torch.load(model_path))
param_dict = {
    "train_batch_size": 2,
    "epoch": 15,
}
train_net = TrainNet(
    net, torch.nn.CrossEntropyLoss(), trainset, testset, MODEL_DIR, param_dict
)
train_net.run()
