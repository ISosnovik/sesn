'''MIT License. Copyright (c) 2020 Ivan Sosnovik, Michał Szmaja'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from .impl.scale_modules import XU_SIConv2d


def pool_from_groups(x, num_groups, pool='max'):
    B, C, H, W = x.shape
    x = x.view(B, num_groups, C // num_groups, H, W)
    if pool == 'max':
        x = x.max(1)[0]
    if pool == 'avg':
        x = x.mean(1)
    return x


class MNIST_XU_Pool(nn.Module):

    def __init__(self, pool_size=4, scales=[1], pool='max'):
        super().__init__()
        C1, C2, C3 = 32, 63, 95
        self.num_scales = len(scales)
        self.pool = pool

        self.main = nn.Sequential(
            XU_SIConv2d(1, C1, 7, scales=scales, num_input_scales=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(C1 * len(scales)),

            XU_SIConv2d(C1, C2, 7, scales=scales, num_input_scales=len(scales)),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(C2 * len(scales)),

            XU_SIConv2d(C2, C3, 7, scales=scales, num_input_scales=len(scales)),
            nn.ReLU(True),
            nn.MaxPool2d(pool_size, padding=2),
            nn.BatchNorm2d(C3 * len(scales)),
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
        x = pool_from_groups(x, self.num_scales, self.pool)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


def mnist_xu_28(**kwargs):
    scales = [0.3, 0.45, 0.67, 1.0, 1.49, 2.23, 3.33]
    return MNIST_XU_Pool(pool_size=4, scales=scales, pool='max')


def mnist_xu_56(**kwargs):
    scales = [0.3, 0.45, 0.67, 1.0, 1.49, 2.23, 3.33]
    return nn.Sequential(nn.Upsample(scale_factor=2), MNIST_XU_Pool(pool_size=8, scales=scales, pool='max'))
