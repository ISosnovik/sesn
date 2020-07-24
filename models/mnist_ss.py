'''It is a modified version of the official implementation of
"Scale-steerable filters for the locally-scale invariant convolutional neural network"
Paper: https://arxiv.org/pdf/1906.03861.pdf
Code: https://github.com/rghosh92/SS-CNN

MIT License
Copyright (c) 2020 Ivan Sosnovik, Michał Szmaja
'''
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import numpy as np

from .impl.scale_steerable import *


class MNIST_SS(nn.Module):
    def __init__(self, pool_size=4, ker_size_range=np.arange(7, 19, 2)):
        super().__init__()

        kernel_sizes = [11, 11, 11]
        pads = (np.array(kernel_sizes) - 1) / 2
        pads = pads.astype(int)
        lays = [30, 60, 90]

        self.conv1 = ScaleConv_steering(1, lays[0], [kernel_sizes[0], kernel_sizes[0]], 1,
                                        padding=pads[0], sigma_phi_range=[np.pi / 16],
                                        k_range=[0.5, 1, 2], ker_size_range=ker_size_range,
                                        phi_range=np.linspace(0, np.pi, 9),
                                        phase_range=[-np.pi / 4],
                                        mode=1)
        self.conv2 = ScaleConv_steering(lays[0], lays[1], [kernel_sizes[1], kernel_sizes[1]], 1, padding=pads[1],
                                        k_range=[0.5, 1, 2], sigma_phi_range=[np.pi / 16],
                                        ker_size_range=ker_size_range,
                                        phi_range=np.linspace(0, np.pi, 9),
                                        phase_range=[-np.pi / 4],
                                        mode=1, drop_rate=2)
        self.conv3 = ScaleConv_steering(lays[1], lays[2], [kernel_sizes[2], kernel_sizes[2]], 1, padding=pads[2],
                                        k_range=[0.5, 1, 2], sigma_phi_range=[np.pi / 16],
                                        phase_range=[-np.pi / 4],
                                        phi_range=np.linspace(0, np.pi, 9),
                                        ker_size_range=ker_size_range,
                                        mode=1, drop_rate=4)

        self.pool1 = nn.MaxPool2d(2)
        self.bn1 = nn.BatchNorm2d(lays[0])
        self.pool2 = nn.MaxPool2d(2)
        self.bn2 = nn.BatchNorm2d(lays[1])
        self.pool3 = nn.MaxPool2d(pool_size, padding=2)
        self.bn3 = nn.BatchNorm2d(lays[2])
        self.bn3_mag = nn.BatchNorm2d(lays[2])
        self.fc1 = nn.Conv2d(lays[2] * 4, 256, 1)
        self.fc1bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.7)
        self.fc2 = nn.Conv2d(256, 10, 1)  # FC2

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        xm = self.pool3(x)
        xm = self.bn3_mag(xm)
        xm = xm.view([xm.shape[0], xm.shape[1] * xm.shape[2] * xm.shape[3], 1, 1])
        xm = self.fc1(xm)
        xm = self.relu(self.fc1bn(xm))
        xm = self.dropout(xm)
        xm = self.fc2(xm)
        xm = xm.view(xm.size()[0], xm.size()[1])
        return xm


def mnist_ss_28(**kwargs):
    return MNIST_SS(pool_size=4, ker_size_range=np.arange(5, 15, 2))


def mnist_ss_56(**kwargs):
    return nn.Sequential(nn.Upsample(scale_factor=2), MNIST_SS(pool_size=8, ker_size_range=np.arange(7, 19, 2)))
