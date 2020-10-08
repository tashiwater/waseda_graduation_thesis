#!/usr/bin/env python3
# coding: utf-8
import os
from mydataset import CustomDataSet as MyDataSet
from MTRNN import CustomNet
from train_custom import TrainNet
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = CURRENT_DIR + "/../data/"
TRAIN_PATH = DATA_DIR + "train"
TEST_PATH = DATA_DIR + "test"
MODEL_DIR = DATA_DIR + "customMTRNN/"
# one_sequence_size = 600  # traning dataのデータ数
# trainset = MyDataSet(0, one_sequence_size, 0.02, 3, 0.1)
# testset = MyDataSet(1, one_sequence_size, 0.02, 1)
tactile_frame_num = 5
trainset = MyDataSet(TRAIN_PATH, tactile_frame_num)
testset = MyDataSet(TEST_PATH, tactile_frame_num)
in_size = 46
net = CustomNet(
    layer_size={"in": in_size, "out": in_size, "io": 34, "cf": 160, "cs": 13},
    tau={"tau_io": 2, "tau_cf": 5, "tau_cs": 50},
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
    "dims": [50],
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
# train_net.load_model("20200719_173814_500")
# for data in train_net._net.parameters():
#     print(data)

train_net.run()
