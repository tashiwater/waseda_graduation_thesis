#!/usr/bin/env python3
# coding: utf-8

import torch
import torchvision
import matplotlib.pyplot as plt
import datetime


class TrainNet:
    def __init__(self, net, criterion, trainset, testset, model_dir, param_dict={}):
        # default parameter
        self._param_dict = {
            "train_batch_size": 4,
            "test_batch_size": len(testset),
            "epoch": 30,
            "save_interval": 50,
            "weight_decay": 0.005,
            "learn_rate": 0.0001,
            "momentum": 0.9,
        }

        self._param_dict.update(param_dict)

        self._net = net
        self._trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=self._param_dict["train_batch_size"],
            shuffle=True,
            num_workers=4,
        )
        self._testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=self._param_dict["test_batch_size"],
            shuffle=True,
            num_workers=4,
        )
        self._device = torch.device("cuda:0")
        self._net = self._net.to(self._device)
        print(self._net)
        self._criterion = criterion
        self._optimizer = torch.optim.SGD(
            self._net.parameters(),
            lr=self._param_dict["learn_rate"],
            momentum=self._param_dict["momentum"],
            weight_decay=self._param_dict["weight_decay"],
        )
        self._model_dir = model_dir

    def run(self):
        """TRAINING"""
        self._train_loss_value = []  # training loss list
        self._train_acc_value = []  # training accuracy list
        self._test_loss_value = []  # test loss list
        self._test_acc_value = []  # test accuracy list

        for epoch in range(self._param_dict["epoch"]):
            print("epoch:", epoch + 1)
            self.train_each_epoch()
            self.test_each_epoch()
            if ((epoch + 1) % self._param_dict["save_interval"]) == 0 and epoch != 0:
                self.model_save("_{}".format(epoch + 1))
        self.model_save("_finish")
        self.show_result()

    def model_save(self, other_str=""):
        now = datetime.datetime.now()
        filename = self._model_dir + now.strftime("%Y%m%d_%H%M%S") + other_str + ".pth"
        torch.save(self._net.state_dict(), filename)

    def train_each_epoch(self):
        sum_loss = 0.0  # loss
        sum_correct = 0  # the number of accurate answers

        for (inputs, labels) in self._trainloader:
            inputs, labels = inputs.to(self._device), labels.to(self._device)
            self._optimizer.zero_grad()
            outputs = self._net(inputs)
            loss = self._criterion(outputs, labels)
            loss.backward()
            self._optimizer.step()
            sum_loss += loss.item()
            _, predicted = outputs.max(1)
            sum_correct += (predicted == labels).sum().item()

        mean_loss = sum_loss / len(self._trainloader)
        accuracy = sum_correct / len(self._trainloader.dataset)
        print("train meanloss={}, accuracy={}".format(mean_loss, accuracy))
        self._train_loss_value.append(mean_loss)
        self._train_acc_value.append(accuracy)

    def test_each_epoch(self):
        sum_loss = 0.0  # loss
        sum_correct = 0  # the number of accurate answers
        # test with TEST data
        for (inputs, labels) in self._testloader:
            inputs, labels = inputs.to(self._device), labels.to(self._device)
            self._optimizer.zero_grad()
            outputs = self._net(inputs)
            loss = self._criterion(outputs, labels)
            sum_loss += loss.item()
            _, predicted = outputs.max(1)
            sum_correct += (predicted == labels).sum().item()
        mean_loss = sum_loss / len(self._testloader)
        accuracy = sum_correct / len(self._testloader.dataset)
        print("test meanloss={}, accuracy={}".format(mean_loss, accuracy))
        self._test_loss_value.append(mean_loss)
        self._test_acc_value.append(accuracy)

    def show_result(self):
        """RESULT OUTPUT"""
        plt.figure(figsize=(6, 6))

        plt.plot(range(self._param_dict["epoch"]), self._train_loss_value)
        plt.plot(range(self._param_dict["epoch"]), self._test_loss_value, c="#00ff00")
        plt.xlim(0, self._param_dict["epoch"])
        plt.ylim(0, 2.5)
        plt.xlabel("epoch")
        plt.ylabel("LOSS")
        plt.legend(["train loss", "test loss"])
        plt.title("loss")
        plt.savefig("loss_image.png")
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
