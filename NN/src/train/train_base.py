#!/usr/bin/env python3
# coding: utf-8

import torch
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import os


class TrainBase:
    def __init__(
        self,
        net,
        criterion,
        trainset,
        testset,
        model_dir,
        param_dict={},
        load_path="",
        device=None,
    ):
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
        self._device = device
        if device is not None:
            self._net.to(device)
        print("num of (train,test)=({},{})".format(len(trainset), len(testset)))
        self.set_data_loader(trainset, testset)
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

        if load_path != "":
            self.load_model(load_path)

        torch.backends.cudnn.benchmark = True
        self.init()

    def set_data_loader(self, trainset, testset):
        self._trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=self._param_dict["train_batch_size"],
            shuffle=True,
            num_workers=os.cpu_count(),
        )
        self._testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=self._param_dict["test_batch_size"],
            shuffle=False,
            num_workers=os.cpu_count(),
        )

    def init(self):
        pass

    def load_model(self, filename):

        model_path = self._model_dir + filename + ".pth"
        csv_path = self._model_dir + filename + ".csv"
        checkpoint = torch.load(model_path)
        self._net.load_state_dict(checkpoint["model"])
        self._optimizer.load_state_dict(checkpoint["optimizer"])
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
        while self._param_dict["epoch"] is None or epoch < self._param_dict["epoch"]:
            epoch += 1
            print("epoch:", epoch)
            self._net.train()
            train_mean_loss = self._each_epoch("train", self._trainloader)
            if (epoch % self._param_dict["graph_span"]) == 0 or epoch == 1:
                self._net.eval()
                test_mean_loss = self._each_epoch("test", self._testloader)
                self._log_loss(epoch, train_mean_loss, test_mean_loss)
            if (epoch % self._param_dict["save_span"]) == 0 or epoch == 1:
                self.model_save("_{}".format(epoch))
        self.model_save("_{}finish".format(epoch))
        self._net.eval()
        test_mean_loss = self._each_epoch("test", self._testloader)
        self._log_loss(epoch, train_mean_loss, test_mean_loss)
        self.show_result()

    def _each_epoch(self, mode, dataloader):
        sum_loss = 0.0  # loss
        calc_num = 0

        for (one_batch_inputs, one_batch_labels) in dataloader:
            motor, tactile, img = one_batch_inputs
            motor = motor.transpose(1, 0)
            tactile = tactile.transpose(1, 0)
            img = img.transpose(1, 0)
            # inputs_transposed = one_batch_inputs.transpose(1, 0)
            labels_transposed = one_batch_labels.transpose(1, 0)
            self._net.init_state(labels_transposed.shape[1])
            outputs = torch.zeros_like(labels_transposed)
            self._optimizer.zero_grad()
            for i, inputs_t in enumerate(zip(motor, tactile, img)):
                outputs[i] = self._net(inputs_t[0], inputs_t[1], inputs_t[2])
            d1 = 0
            loss = []
            for dim, rate in zip(
                self._param_dict["dims"], self._param_dict["loss_rates"]
            ):
                d2 = d1 + dim
                one_loss = self._criterion(
                    outputs[:, :, d1:d2], labels_transposed[:, :, d1:d2]
                )
                loss.append(one_loss * rate)
                d1 = d2

            # loss = self._criterion(outputs, labels_transposed)
            loss = sum(loss)
            if mode == "train":
                # sum(loss).backward()
                # pritn(self._net.cs2cs.parameters.grad)
                loss.backward()
                self._optimizer.step()
            # sum_loss += sum(loss).item()
            sum_loss += loss.item()
            calc_num += 1
        mean_loss = sum_loss / calc_num
        print(mode + " meanloss={}".format(mean_loss))
        return mean_loss

    def model_save(self, other_str=""):
        now = datetime.datetime.now()
        filename = self._model_dir + now.strftime("%Y%m%d_%H%M%S") + other_str
        state = {
            "model": self._net.state_dict(),
            "optimizer": self._optimizer.state_dict(),
        }
        torch.save(state, filename + ".pth", _use_new_zipfile_serialization=False)

        df = pd.DataFrame(
            data={
                "epoch": self._graph_epoch,
                "train_loss": self._train_loss_value,
                "test_loss": self._test_loss_value,
            },
            columns=["epoch", "train_loss", "test_loss"],
        )
        df.to_csv(filename + ".csv", index=False)
        print("save " + filename)
        if len(self._graph_epoch) > 10:
            self.show_result()

    def show_result(self):
        """RESULT OUTPUT"""

        plt.figure(figsize=(6, 6))
        plt.plot(self._graph_epoch[10:], self._train_loss_value[10:])
        plt.plot(self._graph_epoch[10:], self._test_loss_value[10:], c="#00ff00")
        plt.xlim(1, self._param_dict["epoch"])
        # # plt.ylim(0, 0.2)
        plt.xlabel("epoch")
        plt.ylabel("LOSS")
        plt.legend(["train loss", "test loss"])
        plt.title("loss")
        plt.savefig(self._model_dir + "loss_image.png")
        plt.clf()

    def _log_loss(self, epoch, train_mean_loss, test_mean_loss):
        self._graph_epoch.append(epoch)
        self._train_loss_value.append(train_mean_loss)
        self._test_loss_value.append(test_mean_loss)
