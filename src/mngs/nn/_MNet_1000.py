#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2023-04-23 09:48:20 (ywatanabe)"

#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import mngs

MNet_config = {
    "classes": ["class1", "class2"],
    "n_chs": 270,
    "n_fc1": 1024,
    "d_ratio1": 0.85,
    "n_fc2": 256,
    "d_ratio2": 0.85,
}


class MNet_1000(nn.Module):
    def __init__(self, config):
        super().__init__()

        # basic
        self.config = config

        # conv
        self.conv1 = nn.Conv2d(1, 40, kernel_size=(config["n_chs"], 4))
        self.act1 = nn.Mish()

        self.conv2 = nn.Conv2d(40, 40, kernel_size=(1, 4))
        self.bn2 = nn.BatchNorm2d(40)
        self.pool2 = nn.MaxPool2d((1, 5))
        self.act2 = nn.Mish()

        self.swap = SwapLayer()

        self.conv3 = nn.Conv2d(1, 50, kernel_size=(8, 12))
        self.bn3 = nn.BatchNorm2d(50)
        self.pool3 = nn.MaxPool2d((3, 3))
        self.act3 = nn.Mish()

        self.conv4 = nn.Conv2d(50, 50, kernel_size=(1, 5))
        self.bn4 = nn.BatchNorm2d(50)
        self.pool4 = nn.MaxPool2d((1, 2))
        self.act4 = nn.Mish()

        # fc
        n_fc_in = 15950

        self.fc = nn.Sequential(
            nn.Linear(n_fc_in, config["n_fc1"]),
            nn.Mish(),
            nn.Dropout(config["d_ratio1"]),
            nn.Linear(config["n_fc1"], config["n_fc2"]),
            nn.Mish(),
            nn.Dropout(config["d_ratio2"]),
            nn.Linear(config["n_fc2"], len(config["classes"])),
        )

    @staticmethod
    def _reshape_input(x, n_chs):
        """
        (batch, channel, time_length) -> (batch, channel, time_length, new_axis)
        """
        if x.ndim == 3:
            x = x.unsqueeze(-1)
        if x.shape[2] == n_chs:
            x = x.transpose(1, 2)
        x = x.transpose(1, 3).transpose(2, 3)
        return x

    @staticmethod
    def _znorm_along_time(x):
        return (x - x.mean(dim=-1, keepdims=True)) / x.std(dim=-1, keepdims=True)

    def forward(self, x):
        # time-wise normalization
        x = self._znorm_along_time(x)
        x = self._reshape_input(x, self.config["n_chs"])

        x = self.act1(self.conv1(x))
        x = self.act2(self.pool2(self.bn2(self.conv2(x))))
        x = self.swap(x)

        x = self.act3(self.pool3(self.bn3(self.conv3(x))))
        x = self.act4(self.pool4(self.bn4(self.conv4(x))))

        x = x.reshape(len(x), -1)

        x = self.fc(x)
        return x


class SwapLayer(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x):
        return x.transpose(1, 2)


if __name__ == "__main__":
    ## Demo data
    BS, N_CHS, SEQ_LEN = 16, 270, 1000
    x = torch.rand(BS, N_CHS, SEQ_LEN).cuda()

    ## Config for the model
    model = MNet_1000(MNet_config).cuda()

    y = model(x)
    summary(model, x)
    print(y.shape)
