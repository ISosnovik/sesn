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

from .impl.ses_conv import SESConv_H_H, SESConv_Z2_H, SESConv_H_H_1x1, SESMaxProjection


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, scales=[1.0], pool=False, interscale=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm3d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        if pool:
            self.conv1 = nn.Sequential(
                SESMaxProjection(),
                SESConv_Z2_H(in_planes, out_planes, kernel_size=7, effective_size=3,
                             stride=stride, padding=3, bias=False, scales=scales, basis_type='A')
            )
        else:
            if interscale:
                self.conv1 = SESConv_H_H(in_planes, out_planes, 2, kernel_size=7, effective_size=3, stride=stride,
                                         padding=3, bias=False, scales=scales, basis_type='A')
            else:
                self.conv1 = SESConv_H_H(in_planes, out_planes, 1, kernel_size=7, effective_size=3, stride=stride,
                                         padding=3, bias=False, scales=scales, basis_type='A')
        self.bn2 = nn.BatchNorm3d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = SESConv_H_H(out_planes, out_planes, 1, kernel_size=7, effective_size=3, stride=1,
                                 padding=3, bias=False, scales=scales, basis_type='A')
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and SESConv_H_H_1x1(in_planes, out_planes,
                                                                      stride=stride, bias=False, num_scales=len(scales)) or None

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
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, scales=[0.0], pool=False, interscale=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes,
                                      nb_layers, stride, dropRate, scales, pool, interscale)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, scales, pool, interscale):
        layers = []
        for i in range(nb_layers):
            pool_layer = pool and (i == 0)
            interscale_layer = interscale and (i == 0)
            layers.append(block(i == 0 and in_planes or out_planes,
                                out_planes, i == 0 and stride or 1, dropRate, scales,
                                pool=pool_layer, interscale=interscale_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0, scales=[1.0],
                 pools=[False, False, False], interscale=[False, False, False]):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = SESConv_Z2_H(3, nChannels[0], kernel_size=7, effective_size=3, stride=1,
                                  padding=3, bias=False, scales=scales, basis_type='A')
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1,
                                   dropRate, scales=scales, pool=pools[0], interscale=interscale[0])
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2,
                                   dropRate, scales=scales, pool=pools[1], interscale=interscale[1])
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2,
                                   dropRate, scales=scales, pool=pools[2], interscale=interscale[2])
        # global average pooling and classifier
        self.proj = SESMaxProjection()
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, (SESConv_H_H, SESConv_Z2_H, SESConv_H_H_1x1)):
                nelement = m.weight.nelement()
                n = nelement / m.in_channels
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.proj(out)
        out = self.relu(self.bn1(out))

        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, self.nChannels)
        out = self.fc(out)
        return out


def wrn_16_8_ses_a(num_classes, **kwargs):
    scales = [0.9 * 1.41**i for i in range(3)]
    return WideResNet(depth=16, num_classes=10, widen_factor=8, dropRate=0.3, scales=scales, pools=[False, False, False])


def wrn_16_8_ses_b(num_classes, **kwargs):
    scales = [0.9 * 1.41**i for i in range(3)]
    return WideResNet(depth=16, num_classes=10, widen_factor=8, dropRate=0.3, scales=scales, pools=[False, True, True])


def wrn_16_8_ses_c(num_classes, **kwargs):
    scales = [1.1 * 1.41**i for i in range(3)]
    return WideResNet(depth=16, num_classes=10, widen_factor=8, dropRate=0.3, scales=scales,
                      pools=[False, False, False], interscale=[True, False, False])
