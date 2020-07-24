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

from .impl.scale_steerable import ScaleConv_steering


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, size_range=[3]):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = ScaleConv_steering(in_planes, out_planes, [3, 3], stride=stride,
                                        padding=1, k_range=[1], ker_size_range=size_range, relu=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = ScaleConv_steering(out_planes, out_planes, [3, 3], padding=1,
                                        k_range=[1], ker_size_range=size_range, relu=False)

        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

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
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, size_range=[3]):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes,
                                      nb_layers, stride, dropRate, size_range)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, size_range):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes,
                                out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0, size_range=[3]):
        super(WideResNet, self).__init__()
        nChannels = [11, 11 * widen_factor, 22 * widen_factor, 44 * widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = ScaleConv_steering(3, nChannels[0], [3, 3], padding=1, k_range=[1],
                                        ker_size_range=size_range, relu=False)

        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, size_range)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate, size_range)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate, size_range)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))

        out = F.avg_pool2d(out, 24)
        out = out.view(-1, self.nChannels)
        out = self.fc(out)
        return out


def wrn_16_8_ss(num_classes, **kwargs):
    size_range = [3, 5, 7]
    return WideResNet(depth=16, num_classes=10, widen_factor=8, dropRate=0.3, size_range=size_range)
