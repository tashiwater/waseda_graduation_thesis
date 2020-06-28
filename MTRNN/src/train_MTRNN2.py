#!/usr/bin/env python3
# coding: utf-8

import torch
import torchvision
import matplotlib.pyplot as plt
import datetime
import pandas as pd


class TrainNet:
    def __init__(self, net, criterion, trainset, testset, model_dir, param_dict={}):
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
        print("num of (train,test)=({},{})".format(len(trainset), len(testset)))
        self._trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=self._param_dict["train_batch_size"],
            shuffle=False,
            num_workers=4,
        )
        self._testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=self._param_dict["test_batch_size"],
            shuffle=False,
            num_workers=4,
        )
        self._device = torch.device("cuda:0")
        self._net = self._net.to(self._device)
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

    def load_model(self, filename):

        model_path = self._model_dir + filename + ".pth"
        csv_path = self._model_dir + filename + ".csv"
        self._net.load_state_dict(torch.load(model_path))
        df = pd.read_csv(csv_path)
        self._graph_epoch = list(df["epoch"])  # x dim of figure
        self._train_loss_value = list(df["train_loss"])
        self._test_loss_value = list(df["test_loss"])  # test loss list

    def run(self):
        """TRAINING"""
        if self._graph_epoch == []:
            epoch = 0
        else:
            epoch = self._graph_epoch[-1]
        while True:
            epoch += 1
            print("epoch:", epoch)
            train_mean_loss = self.train_each_epoch()
            if (epoch % self._param_dict["graph_span"]) == 0 or epoch == 1:
                test_mean_loss = self.test_each_epoch()
                self._graph_epoch.append(epoch)
                self._train_loss_value.append(train_mean_loss)
                self._test_loss_value.append(test_mean_loss)
            if (epoch % self._param_dict["save_span"]) == 0:
                self.model_save("_{}".format(epoch))
            if (
                self._param_dict["epoch"] is not None
                and epoch >= self._param_dict["epoch"]
            ):
                break

        self.model_save("_{}finish".format(epoch))
        test_mean_loss = self.test_each_epoch()
        self._graph_epoch.append(epoch)
        self._train_loss_value.append(train_mean_loss)
        self._test_loss_value.append(test_mean_loss)
        self.show_result()

    def model_save(self, other_str=""):
        now = datetime.datetime.now()
        filename = self._model_dir + now.strftime("%Y%m%d_%H%M%S") + other_str
        torch.save(self._net.state_dict(), filename + ".pth")

        df = pd.DataFrame(
            data={
                "epoch": self._graph_epoch,
                "train_loss": self._train_loss_value,
                "test_loss": self._test_loss_value,
            },
            columns=["epoch", "train_loss", "test_loss"],
        )
        df.to_csv(filename + ".csv")
        print("save " + filename)

    def train_each_epoch(self):
        self._net.train()
        sum_loss = 0.0  # loss
        calc_num = 0
        for (one_batch_inputs, one_batch_labels) in self._trainloader:
            for inputs2, labels2 in zip(one_batch_inputs, one_batch_labels):
                hidden_state = self._net._get_initial_hidden_states(inputs2[0].size(0))
                outputs = torch.empty_like(labels2)
                self._optimizer.zero_grad()
                for i, (data, label) in enumerate(zip(inputs2, labels2)):
                    data, label, = data.to(self._device), label.to(self._device)
                    hidden_state = [hidden.to(self._device) for hidden in hidden_state]
                    outputs[i], hidden_state = self._net(data, hidden_state)
                loss = self._criterion(outputs, labels2)
                loss.backward()
                self._optimizer.step()
                sum_loss += loss.item()
                calc_num += 1

        mean_loss = sum_loss / calc_num
        print("train meanloss={}".format(mean_loss))
        return mean_loss

    def test_each_epoch(self):
        self._net.eval()
        sum_loss = 0.0  # loss
        # calc_num = 0
        # for (inputs, labels) in self._testloader:
        #     hidden_state = self._net._get_initial_hidden_states(inputs[0].size(0))
        #     for data, label in zip(inputs, labels):
        #         data, label, = data.to(self._device), label.to(self._device)
        #         hidden_state = [hidden.to(self._device) for hidden in hidden_state]
        #         output, new_hidden_state = self._net(data, hidden_state)
        #         loss = self._criterion(output, label)
        #         hidden_state = new_hidden_state
        #         sum_loss += loss.item()
        #         calc_num += 1
        # mean_loss = sum_loss / calc_num
        # print("test meanloss={}".format(mean_loss))
        return 0  # mean_loss

    def show_result(self):
        """RESULT OUTPUT"""

        plt.figure(figsize=(6, 6))
        plt.plot(self._graph_epoch, self._train_loss_value)
        plt.plot(self._graph_epoch, self._test_loss_value, c="#00ff00")
        plt.xlim(1, self._param_dict["epoch"])
        # # plt.ylim(0, 0.2)
        plt.xlabel("epoch")
        plt.ylabel("LOSS")
        plt.legend(["train loss", "test loss"])
        plt.title("loss")
        plt.savefig("loss_image.png")
        """
        plt.clf()
        plt.plot(range(self._param_dict["epoch"]), self._train_acc_value)
        plt.plot(range(self._param_dict["epoch"]), self._test_acc_value, c="#00ff00")
        plt.xlim(0, self._param_dict["epoch"])
        plt.ylim(0, 1)
        plt.xlabel("epoch")
        plt.ylabel("ACCURACY")
        plt.legend(["train acc", "test acc"])
        plt.title("accuracy")
        plt.savefig("accuracy_image.png")
"""
