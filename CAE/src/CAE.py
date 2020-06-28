#!/usr/bin/env python3
# coding: utf-8
import torch


class ToImg(torch.nn.Module):
    def forward(self, x):
        n, _ = x.shape
        return x.reshape(n, 256, 6, 8)


class Nothing(torch.nn.Module):
    def __init__(self, num):
        super().__init__()
        self._str = str(num)

    def forward(self, x):
        print(self._str)
        return x


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        hidden_dim = 20
        self.encoder = torch.nn.Sequential(
            # 128*96*3
            # Nothing(1),
            torch.nn.Conv2d(3, 32, 4, stride=2, padding=1),  # ->64*48*32
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 4, stride=2, padding=1),  # ->32*24*64
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 4, stride=2, padding=1),  # ->16*12*128
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, 4, stride=2, padding=1),  # ->8*6*256
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            # Nothing(2),
            torch.nn.Linear(8 * 6 * 256, 254),
            torch.nn.BatchNorm1d(254),
            torch.nn.ReLU(),
            # Nothing(3),
            torch.nn.Linear(254, hidden_dim),
            # Nothing(4),
            torch.nn.BatchNorm1d(hidden_dim),
            # Nothing(5),
            torch.nn.Sigmoid(),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 254),
            torch.nn.BatchNorm1d(254),
            torch.nn.ReLU(),
            torch.nn.Linear(254, 8 * 6 * 256),
            torch.nn.BatchNorm1d(8 * 6 * 256),
            torch.nn.ReLU(),
            ToImg(),
            torch.nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        image = x
        image2 = self.encoder(image)
        x = self.decoder(image2)

        return x


if __name__ == "__main__":
    net = Net()
    print(net)

"""

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        hidden_dim = 40
        self.encoder = torch.nn.Sequential(
            # 128*96*3
            torch.nn.Conv2d(3, 32, 4, stride=2, padding=1),  # ->64*48*32
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 4, stride=2, padding=1),  # ->32*24*64
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 4, stride=2, padding=1),  # ->16*12*128
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, 4, stride=2, padding=1),  # ->8*6*256
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(8 * 6 * 256, 254),
            torch.nn.BatchNorm1d(254),
            torch.nn.ReLU(),
            torch.nn.Linear(254, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.Sigmoid(),
        )
        self.decode_linear = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 254),
            torch.nn.BatchNorm1d(254),
            torch.nn.ReLU(),
            torch.nn.Linear(254, 8 * 6 * 256),
            torch.nn.BatchNorm1d(8 * 6 * 256),
            torch.nn.ReLU(),
        )
        self.decoder = torch.nn.Sequential(
            # torch.nn.Linear(hidden_dim, 254),
            # torch.nn.BatchNorm1d(254),
            # torch.nn.ReLU(),
            # torch.nn.Linear(254, 8 * 6 * 256),
            # torch.nn.BatchNorm1d(8 * 6 * 256),
            # torch.nn.ReLU(),
            # ToImg(),
            torch.nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        images = x
        x = self.encoder(x)
        x = self.decode_linear(x)
        x = x.view([-1, 256, 6, 8])
        x = self.decoder(x)
        return x
        self.bn7 = torch.nn.BatchNorm1d(20)


            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 12),
            nn.ReLU(True),
            nn.Linear(12, 2))

        
        self.relu = torch.nn.ReLU()

        self.conv1 = torch.nn.Conv2d(3, 32, 4, stride=2, padding=1)
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.conv2 = torch.nn.Conv2d(32, 64, 4, stride=2, padding=1)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.conv3 = torch.nn.Conv2d(64, 128, 3, stride=2, padding=1)
        torch.nn.init.kaiming_normal_(self.conv3.weight)
        self.bn3 = torch.nn.BatchNorm2d(128)
        self.conv4 = torch.nn.Conv2d(128, 256, 4, stride=2, padding=1)
        torch.nn.init.kaiming_normal_(self.conv4.weight)
        self.bn4 = torch.nn.BatchNorm2d(256)
        # self.conv3 = nn.Conv2d(16, 24, 5)

        self.fc1 = torch.nn.Linear(8*6*256, 254)
        self.bn5 = torch.nn.BatchNorm1d(254)
        self.fc2 = torch.nn.Linear(254, 20)
        self.bn6 = torch.nn.BatchNorm1d(20)
        self.fc3 = torch.nn.Linear( 20,254)
        self.bn7 = torch.nn.BatchNorm1d(20)
        
        
        
        
        self.fc2 = torch.nn.Linear(1000, 120)
        self.fc3 = torch.nn.Linear(120, 60)
        self.fc4 = torch.nn.Linear(60, 4)  # 3 classes

    def encode(self, x):
        # x:128*96*3
        x = self.conv1(x)  # ->64*48*32
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)  # ->32*24*64
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)  # ->16*12*128
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)  # ->8*6*256
        x = self.bn4(x)
        x = self.relu(x)

        x = self.pool(x)  # ->29*21*16
        # x = self.conv3(x) #->
        # x = self.relu(x)
        # x = self.pool(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
"""
