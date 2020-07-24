'''This file contains our implemetation of the following papers:
- Locally Scale-Invariant Convolutional Layer 
    https://arxiv.org/pdf/1412.5104.pdf

- Scale-Invariant Convolutional Layer 
    https://arxiv.org/abs/1411.6369

MIT License
Copyright (c) 2020 Ivan Sosnovik, Michał Szmaja
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import math


def rescale4d(x, scale, mode='bilinear', padding_mode='constant'):
    """rescales 4D tensor while preserving its shape
    Args:
        x: Input 4D tensor of shape [Any, Any, H, W]
        scale: scale factor. scale < 1 stands for downscaling
        mode: interpolation mode for rescaling 
            'nearest' | 'linear' | 'bilinear' | 'bicubic' |  'trilinear' | 'area'
        padding_mode: interpolation mode for padding if needed.
            'constant' | 'reflect' | 'replicate' | 'circular'
    """
    if mode == 'nearest':
        align_corners = None
    else:
        align_corners = True

    if scale == 1:
        return x

    rescaled_x = F.interpolate(x, scale_factor=scale, mode=mode, align_corners=align_corners)

    _, _, H, W = x.shape
    _, _, h, w = rescaled_x.shape
    pad_l = (W - w) // 2
    pad_r = W - w - pad_l
    pad_t = (H - h) // 2
    pad_b = H - h - pad_t
    rescaled_x = F.pad(rescaled_x, (pad_l, pad_r, pad_t, pad_b), mode=padding_mode)
    return rescaled_x


def batchify(x):
    # [N, M, C, H, W] -> [N*M, C, H, W]
    shape = list(x.shape)
    shape[0] *= shape[1]
    del shape[1]
    x = x.view(shape)
    return x


def unbatchify(x, size):
    # [size*B, C, H, W] -> [size, B, C, H, W]
    shape = list(x.shape)
    shape = [size, -1] + shape[1:]
    x = x.view(shape)
    return x


class Kanazawa_SIConv2d(nn.Conv2d):
    '''Locally Scale-Invariant Convolutional Layer. 
    Paper: https://arxiv.org/pdf/1412.5104.pdf
    Args:
        in_channels: Number of channels in the input image
        out_channels: Number of channels produced by the convolution
        kernel_size: Size of the convolving kernel
        scales: list of scales
        stride: Stride of the convolution
        padding: Zero-padding added to both sides of the input
        dilation: Spacing between kernel elements
        groups: Number of blocked connections from input channels to output channels
        bias: If ``True``, adds a learnable bias to the output
        scaling_mode: interpolation mode for rescaling
        scale_padding_mode: interpolation mode for padding if needed.
    '''

    def __init__(self, in_channels, out_channels, kernel_size, scales=[1], stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 scaling_mode='bilinear', scale_padding_mode='constant'):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.scales = scales
        self.scaling_mode = scaling_mode
        self.scale_padding_mode = scale_padding_mode

    def forward(self, x):
        # rescaling
        stack = []
        for scale in self.scales:
            x_rescaled = rescale4d(x, scale, self.scaling_mode, self.scale_padding_mode)
            stack.append(x_rescaled)
        x = torch.stack(stack, 0)
        x = batchify(x)

        # convolve
        x = super().forward(x)

        # rescaling back
        x = unbatchify(x, len(self.scales)).unbind(0)
        output_stack = []
        for x_, scale in zip(x, self.scales):
            x_rescaled = rescale4d(x_, 1 / scale, self.scaling_mode, self.scale_padding_mode)
            output_stack.append(x_rescaled)
        y = torch.stack(output_stack, 0)
        # pool
        y = y.max(0)[0]
        return y


class XU_SIConv2d(nn.Conv2d):
    '''Scale-Invariant Convolutional Layer
    Paper: https://arxiv.org/abs/1411.6369

    Instead of using multiple columns as in the original paper, we utilize `groups` argument in PyTorch Conv2d
    see paper "Deep Roots: Improving CNN Efficiency with Hierarchical Filter Groups"
    https://arxiv.org/pdf/1605.06489.pdf

    Args:
        in_channels: number of channels of the input Tensor
        out_channels: number of channels for the output Tensor
        kernel_size: spatial size of convolutional kernel
        scales: [s1, s2, s3, ...] list of scales. scale greater than 1 stands for upscaling
        stride: Stride of the convolution
        pad_if_needed: True/False whether to pad the tensor or not before the convolution 
        num_input_scales: number of scales of the input tensor (1 or `len(scales)`)

    Example:

        class Model(nn.Module):

            def __init__(self, scales):
                super().__init__()
                self.scales = scales
                self.conv = nn.Sequential(
                    XU_SIConv2d(1, 6, 5, scales, pad_if_needed=True, num_input_scales=1),
                    nn.BatchNorm2d(6 * len(scales)),
                    nn.ReLU(True),
                    nn.MaxPool2d(2, 2),
                    XU_SIConv2d(6, 12, 5, scales, pad_if_needed=True, num_input_scales=3),
                    nn.BatchNorm2d(12 * len(scales)),
                    nn.ReLU(True)
                )

                self.linear = nn.Linear(12 * len(scales), 10)

            def forward(self, x):
                x = self.conv(x)
                x = x.mean(-1).mean(-1) # adaptive avg pooling
                x = self.linear(x)
                return x
    '''

    def __init__(self, in_channels, out_channel, kernel_size, scales=[1], stride=1, pad_if_needed=True, num_input_scales=1):
        if num_input_scales > 1:
            assert len(scales) == num_input_scales
        if pad_if_needed and kernel_size > 1:
            max_size = int(kernel_size * max(scales))
            if kernel_size > 1 and not max_size % 2 == 1:
                print('WARNING: rescaled kernel has even size {}'.format(max_size))
            padding = max_size // 2
        else:
            padding = 0

        super().__init__(in_channels, out_channel, kernel_size, stride=stride, padding=padding, bias=False)
        self.scales = scales
        self.num_input_scales = num_input_scales

    def _get_transformed_kernel(self):
        C_out, C_in, H, W = self.weight.shape
        kernel_size = self.kernel_size[0]

        if kernel_size == 1:
            # we consider 1x1 convolution as in NiN original paper.
            # Which means that its spatial resolution is 0,
            # i.e. a fully-connected layer for feature channels
            kernel = self.weight.unsqueeze(0)
            kernel = kernel.expand(len(self.scales), -1, -1, -1, -1).contiguous()
            kernel = kernel.view(-1, C_in, H, W)
            return kernel

        max_size = int(kernel_size * max(self.scales))
        pad = math.ceil((max_size - kernel_size) / 2)
        kernel = F.pad(self.weight, (pad, pad, pad, pad))
        kernel_norm = torch.norm(kernel, p=1, dim=(2, 3), keepdim=True)
        stack = []
        for scale in self.scales:
            if scale > 1:
                # upscaling is an ill-posed problem. in the original paper,
                # the upscaling method is chosen to give solution with smallest l2 norm.
                # For the case of `bilinear` donwscaling, it is `nearest` upscaling
                mode = 'nearest'
            else:
                mode = 'bilinear'
            rescaled_kernel = rescale4d(kernel, scale, mode=mode)
            rescaled_kernel_norm = torch.norm(rescaled_kernel, p=1, dim=(2, 3), keepdim=True)
            rescaled_kernel = rescaled_kernel * kernel_norm / rescaled_kernel_norm
            stack.append(rescaled_kernel)
        kernel = torch.cat(stack)
        return kernel

    def forward(self, x):
        kernel = self._get_transformed_kernel()
        return F.conv2d(x, kernel, stride=self.stride, padding=self.padding, groups=self.num_input_scales)
