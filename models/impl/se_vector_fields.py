'''It is a reimplementation of "Scale equavariant CNNs with vector fields"
Paper: https://arxiv.org/pdf/1807.11783.pdf
Code: https://github.com/dmarcosg/ScaleEqNet

This reimplementation is slightly faster than the original one

MIT License 
Copyright (c) 2020 Ivan Sosnovik
'''
import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np


class ScaleConvScalar(nn.Module):
    '''Scalar to Vector fields scale-convolution'''

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 n_scales_small=5, n_scales_big=3, angle_range=2 * np.pi / 3, base=1.26):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        log_scales = np.linspace(-n_scales_small + 1, n_scales_big, n_scales_small + n_scales_big)
        self.scales = base ** log_scales
        self.angles = log_scales * angle_range / (n_scales_small + n_scales_big - 1)

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels,
                                                kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x):
        x = [conv_scale(x, self.weight, s, self.padding, self.stride) for s in self.scales]
        vals, args = torch.stack(x, 2).max(2)
        angles = torch.Tensor(self.angles)[args].to(args.device)
        return F.relu(vals) * angles.cos(), F.relu(vals) * angles.sin()


class ScaleConvVector(nn.Module):
    '''Vector to Vector fields scale-convolution'''

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 n_scales_small=5, n_scales_big=3, angle_range=2 * np.pi / 3, base=1.26):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        log_scales = np.linspace(-n_scales_small + 1, n_scales_big, n_scales_small + n_scales_big)
        self.scales = base ** log_scales
        self.angles = log_scales * angle_range / (n_scales_small + n_scales_big - 1)

        self.weight_u = nn.Parameter(torch.Tensor(out_channels, in_channels,
                                                  kernel_size, kernel_size))
        self.weight_v = nn.Parameter(torch.Tensor(out_channels, in_channels,
                                                  kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight_u, a=5**0.5)
        nn.init.kaiming_uniform_(self.weight_v, a=5**0.5)

    def forward(self, u, v):
        outputs = []
        for scale, angle in zip(self.scales, self.angles):
            weight_u = self.weight_u * np.cos(angle) + self.weight_v * np.sin(angle)
            weight_v = -self.weight_u * np.sin(angle) + self.weight_v * np.cos(angle)
            u_out = conv_scale(u, weight_u, scale, self.padding, self.stride)
            v_out = conv_scale(v, weight_v, scale, self.padding, self.stride)
            outputs.append(u_out + v_out)
        #
        vals, args = torch.stack(outputs, 2).max(2)
        angles = torch.Tensor(self.angles)[args].to(args.device)
        return F.relu(vals) * angles.cos(), F.relu(vals) * angles.sin()


class VectorBatchNorm(nn.Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.register_buffer('running_var', torch.ones(1, num_features, 1, 1))

    def forward(self, u, v):
        if self.training:
            var = vector2scalar(u, v).var(dim=(0, 2, 3), unbiased=False, keepdims=True)
            n = u.nelement() / u.size(1)
            with torch.no_grad():
                self.running_var *= 1 - self.momentum
                self.running_var += self.momentum * var * n / (n - 1)

        else:
            var = self.running_var

        u = u / (self.eps + var).sqrt()
        v = v / (self.eps + var).sqrt()
        return u, v


class VectorMaxPool(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding

    def forward(self, u, v):
        B, C, _, _ = u.shape
        x = vector2scalar(u, v)
        _, idx = F.max_pool2d_with_indices(x, kernel_size=self.kernel_size,
                                           stride=self.stride, padding=self.padding)
        u = torch.gather(u.view(B, C, -1), 2, idx.view(B, C, -1)).view_as(idx)
        v = torch.gather(v.view(B, C, -1), 2, idx.view(B, C, -1)).view_as(idx)
        return u, v


class VectorDropout(nn.Module):
    '''Dropout with synchronized masks
    '''

    def __init__(self, p=0.5):
        assert p < 1.0
        super().__init__()
        self.p = p

    def forward(self, input):
        u, v = input
        probs = u.data.new(u.data.size()).fill_(1 - self.p)
        mask = torch.bernoulli(probs) / (1 - self.p)
        return u * mask, v * mask


# FUNCTIONS
def vector2scalar(u, v):
    '''Vector field to Scalar field projection
    (u, v) --> sqrt(u**2 + v**2)
    '''
    return (u**2 + v**2)**0.5


def conv_scale(x, weight, scale, padding, stride):
    original_size = x.shape[-1]
    kernel_size = weight.shape[-1]
    output_size = (original_size + 1 + padding * 2 - kernel_size) // stride
    size = int(round(original_size * scale))
    x = F.interpolate(x, size=size, align_corners=False, mode='bilinear')
    x = F.conv2d(x, weight, stride=stride, padding=padding)
    x = F.interpolate(x, size=output_size, align_corners=False, mode='bilinear')
    return x
