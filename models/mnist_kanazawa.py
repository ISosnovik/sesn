'''MIT License. Copyright (c) 2020 Ivan Sosnovik, Michał Szmaja'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from .impl.scale_modules import Kanazawa_SIConv2d


class MNIST_Kanazawa(nn.Module):

    def __init__(self, pool_size=4, scales=[1]):
        super().__init__()
        C1, C2, C3 = 32, 63, 95
        self.main = nn.Sequential(
            Kanazawa_SIConv2d(1, C1, 7, scales=scales, padding=3),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(C1),

            Kanazawa_SIConv2d(C1, C2, 7, scales=scales, padding=3),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(C2),

            Kanazawa_SIConv2d(C2, C3, 7, scales=scales, padding=3),
            nn.ReLU(True),
            nn.MaxPool2d(pool_size, padding=2),
            nn.BatchNorm2d(C3),
        )

        self.linear = nn.Sequential(
            nn.Linear(4 * C3, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.7),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


def mnist_kanazawa_28(**kwargs):
    scales = [0.3, 0.45, 0.67, 1.0, 1.49, 2.23, 3.33]
    return MNIST_Kanazawa(pool_size=4, scales=scales)


def mnist_kanazawa_56(**kwargs):
    scales = [0.3, 0.45, 0.67, 1.0, 1.49, 2.23, 3.33]
    return nn.Sequential(nn.Upsample(scale_factor=2), MNIST_Kanazawa(pool_size=8, scales=scales))
