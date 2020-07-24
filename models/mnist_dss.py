'''MIT License. Copyright (c) 2020 Ivan Sosnovik, Michał Szmaja'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from .impl.deep_scale_space import Dconv2d, BesselConv2d, ScaleMaxProjection


class MNIST_DSS_Vector(nn.Module):

    def __init__(self, pool_size=4, n_scales=4, scale_sizes=[1, 1, 1], init='he'):
        super().__init__()
        C1, C2, C3 = 32, 63, 95
        S1, S2, S3 = scale_sizes
        C1 = int(C1 / S1**0.5)
        if S2 == 1 and S3 == 1:
            C2 = int(C2 * S1**0.5)
            C3 = int(C3 / S1**0.5)
            FC1 = int(256 * S1 ** 0.5)
        else:
            C2 = int(C2 / S2**0.5)
            C3 = int(C3 / S3**0.5)
            FC1 = int(256 * S3 ** 0.5)

        n1 = n_scales
        n2 = n1 - S1 // 2
        n3 = n2 - S2 // 2

        self.main = nn.Sequential(
            BesselConv2d(n_channels=1, base=2, zero_scale=0.25, n_scales=n_scales),
            Dconv2d(1, C1, kernel_size=[S1, 7, 7], base=2,
                    io_scales=[n1, n1], padding=3, init=init),
            nn.ReLU(True),
            nn.MaxPool3d((1, 2, 2)),
            nn.BatchNorm3d(C1),

            Dconv2d(C1, C2, kernel_size=[S2, 7, 7], base=2,
                    io_scales=[n2, n2], padding=3, init=init),
            nn.ReLU(True),
            nn.MaxPool3d((1, 2, 2)),
            nn.BatchNorm3d(C2),

            Dconv2d(C2, C3, kernel_size=[S3, 7, 7], base=2,
                    io_scales=[n3, n3], padding=3, init=init),
            nn.ReLU(True),
            nn.MaxPool3d((1, pool_size, pool_size), padding=(0, 2, 2)),
            nn.BatchNorm3d(C3),
        )

        self.linear = nn.Sequential(
            nn.Linear(4 * C3, FC1, bias=False),
            nn.BatchNorm1d(FC1),
            nn.ReLU(True),
            nn.Dropout(0.7),
            nn.Linear(FC1, 10)
        )

    def forward(self, x):
        x = self.main(x)
        x = ScaleMaxProjection()(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class MNIST_DSS_Scalar(nn.Module):

    def __init__(self, pool_size=4, n_scales=4, scale_sizes=[1, 1, 1], init='he'):
        super().__init__()
        C1, C2, C3 = 32, 63, 95
        S1, S2, S3 = scale_sizes
        C1 = int(C1 / S1**0.5)
        if S2 == 1 and S3 == 1:
            C2 = int(C2 * S1**0.5)
            C3 = int(C3 / S1**0.5)
            FC1 = int(256 * S1 ** 0.5)
        else:
            C2 = int(C2 / S2**0.5)
            C3 = int(C3 / S3**0.5)
            FC1 = int(256 * S3 ** 0.5)

        self.main = nn.Sequential(
            BesselConv2d(n_channels=1, base=2, zero_scale=0.25, n_scales=n_scales),
            Dconv2d(1, C1, kernel_size=[S1, 7, 7], base=2,
                    io_scales=[n_scales, n_scales], padding=3, init=init),
            ScaleMaxProjection(),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(C1),

            BesselConv2d(n_channels=C1, base=2, zero_scale=0.25, n_scales=n_scales),
            Dconv2d(C1, C2, kernel_size=[S2, 7, 7], base=2,
                    io_scales=[n_scales, n_scales], padding=3, init=init),
            ScaleMaxProjection(),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(C2),

            BesselConv2d(n_channels=C2, base=2, zero_scale=0.25, n_scales=n_scales),
            Dconv2d(C2, C3, kernel_size=[S3, 7, 7], base=2,
                    io_scales=[n_scales, n_scales], padding=3, init=init),
            ScaleMaxProjection(),
            nn.ReLU(True),
            nn.MaxPool2d(pool_size, padding=2),
            nn.BatchNorm2d(C3),
        )

        self.linear = nn.Sequential(
            nn.Linear(4 * C3, FC1, bias=False),
            nn.BatchNorm1d(FC1),
            nn.ReLU(True),
            nn.Dropout(0.7),
            nn.Linear(FC1, 10)
        )

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


def mnist_dss_vector_28(**kwargs):
    return MNIST_DSS_Vector(pool_size=4, n_scales=2, scale_sizes=[1, 1, 1])


def mnist_dss_scalar_28(**kwargs):
    return MNIST_DSS_Scalar(pool_size=4, n_scales=2, scale_sizes=[1, 1, 1])


def mnist_dss_vector_56(**kwargs):
    model = MNIST_DSS_Vector(pool_size=8, n_scales=2, scale_sizes=[1, 1, 1])
    return nn.Sequential(nn.Upsample(scale_factor=2), model)


def mnist_dss_scalar_56(**kwargs):
    model = MNIST_DSS_Scalar(pool_size=8, n_scales=2, scale_sizes=[1, 1, 1])
    return nn.Sequential(nn.Upsample(scale_factor=2), model)
