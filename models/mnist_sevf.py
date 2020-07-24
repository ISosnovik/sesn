'''MIT License. Copyright (c) 2020 Ivan Sosnovik, Michał Szmaja'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from .impl.se_vector_fields import *


class MNIST_SEVF_Vector(nn.Module):
    def __init__(self, pool_size=4):
        super(MNIST_SEVF_Vector, self).__init__()
        self.conv1 = ScaleConvScalar(1, 23, 7, padding=3)
        self.pool1 = VectorMaxPool(2)
        self.bn1 = VectorBatchNorm(23)

        self.conv2 = ScaleConvVector(23, 45, 7, padding=3)
        self.pool2 = VectorMaxPool(2)
        self.bn2 = VectorBatchNorm(45)

        self.conv3 = ScaleConvVector(45, 68, 7, padding=3)
        self.pool3 = VectorMaxPool(pool_size, padding=2)

        self.fc1 = nn.Conv2d(68 * 4, 256, 1)  # FC1
        self.fc1bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.7)
        self.fc2 = nn.Conv2d(256, 10, 1)  # FC2

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(*x)
        x = self.bn1(*x)
        x = self.conv2(*x)
        x = self.pool2(*x)
        x = self.bn2(*x)
        x = self.conv3(*x)
        x = self.pool3(*x)

        xm = vector2scalar(*x)
        xm = xm.view(xm.size(0), -1, 1, 1)
        xm = self.fc1(xm)
        xm = self.relu(self.fc1bn(xm))
        xm = self.dropout(xm)
        xm = self.fc2(xm)

        xm = xm.view(xm.size()[0], xm.size()[1])
        return xm


class MNIST_SEVF_Scalar(nn.Module):
    def __init__(self, pool_size=4):
        super(MNIST_SEVF_Scalar, self).__init__()

        self.conv1 = ScaleConvScalar(1, 32, 7, padding=3)
        self.pool1 = nn.MaxPool2d(2)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = ScaleConvScalar(32, 63, 7, padding=3)
        self.pool2 = nn.MaxPool2d(2)
        self.bn2 = nn.BatchNorm2d(63)

        self.conv3 = ScaleConvScalar(63, 95, 7, padding=3)
        self.pool3 = nn.MaxPool2d(pool_size, padding=2)

        self.fc1 = nn.Conv2d(95 * 4, 256, 1)  # FC1
        self.fc1bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.7)
        self.fc2 = nn.Conv2d(256, 10, 1)  # FC2

    def forward(self, x):
        x = vector2scalar(*self.conv1(x))
        x = self.pool1(x)
        x = self.bn1(x)
        x = vector2scalar(*self.conv2(x))
        x = self.pool2(x)
        x = self.bn2(x)
        x = vector2scalar(*self.conv3(x))
        xm = self.pool3(x)
        xm = xm.view(xm.size(0), -1, 1, 1)

        xm = self.fc1(xm)
        xm = self.relu(self.fc1bn(xm))
        xm = self.dropout(xm)
        xm = self.fc2(xm)

        xm = xm.view(xm.size()[0], xm.size()[1])
        return xm


def mnist_sevf_scalar_28(**kwargs):
    return MNIST_SEVF_Scalar(pool_size=4)


def mnist_sevf_scalar_56(**kwargs):
    return nn.Sequential(nn.Upsample(scale_factor=2), MNIST_SEVF_Scalar(pool_size=8))


def mnist_sevf_vector_28(**kwargs):
    return MNIST_SEVF_Vector(pool_size=4)


def mnist_sevf_vector_56(**kwargs):
    return nn.Sequential(nn.Upsample(scale_factor=2), MNIST_SEVF_Vector(pool_size=8))
