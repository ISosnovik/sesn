'''It is a modified version of the unofficial implementaion of 
'Wide Residual Networks'
Paper: https://arxiv.org/abs/1605.07146
Code: https://github.com/xternalz/WideResNet-pytorch

MIT License
Copyright (c) 2020 Ivan Sosnovik, Michał Szmaja
Copyright (c) 2019 xternalz
'''
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math

from .impl.deep_scale_space import Dconv2d, BesselConv2d, ScaleMaxProjection


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, base=2.0, nscales=8, scale_size=1):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm3d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = Dconv2d(in_planes, out_planes, kernel_size=[scale_size, 3, 3],
                             stride=stride, padding=1, base=base, io_scales=[nscales, nscales])
        nscales = nscales - scale_size // 2
        self.bn2 = nn.BatchNorm3d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = Dconv2d(out_planes, out_planes, kernel_size=[1, 3, 3], stride=1,
                             padding=1, io_scales=[nscales, nscales], base=base)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)

        if not self.equalInOut:
            self.convShortcut = Dconv2d(in_planes, out_planes, kernel_size=[1, 1, 1], stride=stride,
                                        padding=0, base=base, io_scales=[nscales, nscales])
        else:
            self.convShortcut = None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, base=2.0, nscales=8, scale_size=1):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate,
                                      base=base, nscales=nscales, scale_size=scale_size)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, base, nscales, scale_size):
        layers = []
        for i in range(nb_layers):
            if not i == 0:
                scale_size = 1
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1,
                                dropRate, base=base, nscales=nscales, scale_size=scale_size))
            nscales -= scale_size // 2
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0,
                 base=2.0, nscales=8, scale_size=2, zero_scale=0.5, init="he"):
        super(WideResNet, self).__init__()
        if scale_size == 1:
            nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        else:
            k = (6 + scale_size) / 7
            k = k**0.5
            C1, C2, C3 = 16, math.ceil(32 / k), 64
            nChannels = [C1, C1 * widen_factor, C2 * widen_factor, C3 * widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock

        self.bessel = BesselConv2d(3, base=base, zero_scale=zero_scale, n_scales=nscales)
        # 1st conv before any network block
        self.conv1 = Dconv2d(3, nChannels[0], kernel_size=[1, 3, 3], stride=1,
                             padding=1, bias=False, io_scales=[nscales, nscales], base=base)

        # nscales -= scale_size // 2
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1,
                                   dropRate, base, nscales, scale_size)
        nscales -= scale_size // 2
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2,
                                   dropRate, base, nscales, scale_size)

        nscales -= scale_size // 2
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2,
                                   dropRate, base, nscales, 1)

        # global average pooling and classifier
        self.bn1 = nn.BatchNorm3d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.proj = ScaleMaxProjection()
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, Dconv2d):
                if init == 'he':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_scales * m.out_channels
                    m.weights.data.normal_(0, math.sqrt(2. / n))
                elif init == 'delta':
                    m.reset_parameters('delta')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.bessel(x)
        out = self.conv1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = self.proj(out)

        out = F.avg_pool2d(out, 24)
        out = out.view(-1, self.nChannels)
        out = self.fc(out)
        return out


def wrn_16_8_dss(num_classes, **kwargs):
    return WideResNet(depth=16, num_classes=10, widen_factor=8, dropRate=0.3,
                      base=2.0, nscales=4, scale_size=2, zero_scale=0.5, init="delta")
