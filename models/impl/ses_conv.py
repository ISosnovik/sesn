'''MIT License. Copyright (c) 2020 Ivan Sosnovik, Michał Szmaja'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ses_basis import steerable_A, steerable_B
from .ses_basis import normalize_basis_by_min_scale


class SESConv_Z2_H(nn.Module):
    '''Scale Equivariant Steerable Convolution: Z2 -> (S x Z2)
    [B, C, H, W] -> [B, C', S, H', W']

    Args:
        in_channels: Number of channels in the input image
        out_channels: Number of channels produced by the convolution
        kernel_size: Size of the convolving kernel
        effective_size: The effective size of the kernel with the same # of params
        scales: List of scales of basis
        stride: Stride of the convolution
        padding: Zero-padding added to both sides of the input
        bias: If ``True``, adds a learnable bias to the output
    '''

    def __init__(self, in_channels, out_channels, kernel_size, effective_size,
                 scales=[1.0], stride=1, padding=0, bias=False, basis_type='A', **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.effective_size = effective_size
        self.scales = [round(s, 3) for s in scales]
        self.num_scales = len(scales)
        self.stride = stride
        self.padding = padding

        if basis_type == 'A':
            basis = steerable_A(kernel_size, scales, effective_size, **kwargs)
        elif basis_type == 'B':
            basis = steerable_B(kernel_size, scales, effective_size, **kwargs)
        basis = normalize_basis_by_min_scale(basis)
        self.register_buffer('basis', basis)

        self.num_funcs = self.basis.size(0)

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, self.num_funcs))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        basis = self.basis.view(self.num_funcs, -1)
        kernel = self.weight @ basis
        kernel = kernel.view(self.out_channels, self.in_channels,
                             self.num_scales, self.kernel_size, self.kernel_size)
        kernel = kernel.permute(0, 2, 1, 3, 4).contiguous()
        kernel = kernel.view(-1, self.in_channels, self.kernel_size, self.kernel_size)

        # convolution
        y = F.conv2d(x, kernel, bias=None, stride=self.stride, padding=self.padding)
        B, C, H, W = y.shape
        y = y.view(B, self.out_channels, self.num_scales, H, W)

        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1, 1)

        return y

    def extra_repr(self):
        s = '{in_channels}->{out_channels} | scales={scales} | size={kernel_size}'
        return s.format(**self.__dict__)


class SESConv_H_H(nn.Module):
    '''Scale Equivariant Steerable Convolution: (S x Z2) -> (S x Z2)
    [B, C, S, H, W] -> [B, C', S', H', W']

    Args:
        in_channels: Number of channels in the input image
        out_channels: Number of channels produced by the convolution
        scale_size: Size of scale filter
        kernel_size: Size of the convolving kernel
        effective_size: The effective size of the kernel with the same # of params
        scales: List of scales of basis
        stride: Stride of the convolution
        padding: Zero-padding added to both sides of the input
        bias: If ``True``, adds a learnable bias to the output
    '''

    def __init__(self, in_channels, out_channels, scale_size, kernel_size, effective_size,
                 scales=[1.0], stride=1, padding=0, bias=False, basis_type='A', **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_size = scale_size
        self.kernel_size = kernel_size
        self.effective_size = effective_size
        self.scales = [round(s, 3) for s in scales]
        self.num_scales = len(scales)
        self.stride = stride
        self.padding = padding

        if basis_type == 'A':
            basis = steerable_A(kernel_size, scales, effective_size, **kwargs)
        elif basis_type == 'B':
            basis = steerable_B(kernel_size, scales, effective_size, **kwargs)

        basis = normalize_basis_by_min_scale(basis)
        self.register_buffer('basis', basis)

        self.num_funcs = self.basis.size(0)

        self.weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels, scale_size, self.num_funcs))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        # get kernel
        basis = self.basis.view(self.num_funcs, -1)
        kernel = self.weight @ basis
        kernel = kernel.view(self.out_channels, self.in_channels, self.scale_size,
                             self.num_scales, self.kernel_size, self.kernel_size)

        # expand kernel
        kernel = kernel.permute(3, 0, 1, 2, 4, 5).contiguous()
        kernel = kernel.view(-1, self.in_channels, self.scale_size,
                             self.kernel_size, self.kernel_size)

        # calculate padding
        if self.scale_size != 1:
            value = x.mean()
            x = F.pad(x, [0, 0, 0, 0, 0, self.scale_size - 1])

        output = 0.0
        for i in range(self.scale_size):
            x_ = x[:, :, i:i + self.num_scales]
            # expand X
            B, C, S, H, W = x_.shape
            x_ = x_.permute(0, 2, 1, 3, 4).contiguous()
            x_ = x_.view(B, -1, H, W)
            output += F.conv2d(x_, kernel[:, :, i], padding=self.padding,
                               groups=S, stride=self.stride)

        # squeeze output
        B, C_, H_, W_ = output.shape
        output = output.view(B, S, -1, H_, W_)
        output = output.permute(0, 2, 1, 3, 4).contiguous()
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1, 1)
        return output

    def extra_repr(self):
        s = '{in_channels}->{out_channels} | scales={scales} | size={kernel_size}'
        return s.format(**self.__dict__)


class SESConv_H_H_1x1(nn.Conv2d):

    def __init__(self, in_channels, out_channel, stride=1, num_scales=1, bias=True):
        super().__init__(in_channels, out_channel, 1, stride=stride, bias=bias)
        self.num_scales = num_scales

    def forward(self, x):
        kernel = self.weight.unsqueeze(0)
        kernel = kernel.expand(self.num_scales, -1, -1, -1, -1).contiguous()
        kernel = kernel.view(-1, self.in_channels, 1, 1)

        B, C, S, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B, -1, H, W)
        x = F.conv2d(x, kernel, stride=self.stride, groups=self.num_scales)

        B, C_, H_, W_ = x.shape
        x = x.view(B, S, -1, H_, W_).permute(0, 2, 1, 3, 4).contiguous()
        return x


class SESMaxProjection(nn.Module):

    def forward(self, x):
        return x.max(2)[0]
