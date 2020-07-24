'''MIT License. Copyright (c) 2020 Ivan Sosnovik, Michał Szmaja'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class MNIST_CNN(nn.Module):

    def __init__(self, pool_size=4):
        super().__init__()
        C1, C2, C3 = 32, 63, 95
        self.main = nn.Sequential(
            nn.Conv2d(1, C1, 7, padding=3),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(C1),

            nn.Conv2d(C1, C2, 7, padding=3),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(C2),

            nn.Conv2d(C2, C3, 7, padding=3),
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


def mnist_cnn_28(**kwargs):
    return MNIST_CNN(pool_size=4)


def mnist_cnn_56(**kwargs):
    return nn.Sequential(nn.Upsample(scale_factor=2), MNIST_CNN(pool_size=8))
