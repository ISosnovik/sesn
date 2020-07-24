'''It is a sligtly modified version of the official implementation of 
"Scale-steerable filters for the locally-scale invariant convolutional neural network"
Paper: https://arxiv.org/pdf/1906.03861.pdf
Code: https://github.com/rghosh92/SS-CNN

MIT License
Copyright (c) 2020 Ivan Sosnovik, Michał Szmaja
'''
import math
import numpy as np
import scipy.ndimage
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter


def generate_filter_basis(filter_size, phi0, sigma, k, scale, phase, drop_rate):
    rot_k = 0
    Mx = (filter_size[0])
    My = (filter_size[1])
    W = np.ones((filter_size[0], filter_size[1]))
    W[np.int((Mx - 1) / 2), np.int((My - 1) / 2)] = 0
    W_dist = scipy.ndimage.morphology.distance_transform_bf(W)

    W_dist[np.int((Mx - 1) / 2), np.int((My - 1) / 2)] = 0
    Mask = np.ones(W_dist.shape)
    Mask[W_dist > np.int((Mx - 1) / 2)] = 0
    W_dist[np.int((Mx - 1) / 2), np.int((My - 1) / 2)] = 1
    W_dist = scale * W_dist
    log_r = np.log(W_dist)

    x_coords = np.zeros((filter_size[0], filter_size[1]))
    y_coords = np.zeros((filter_size[0], filter_size[1]))

    for i in range(x_coords.shape[0]):
        x_coords[i, :] = (((Mx - 1) / 2) - i)

    for i in range(y_coords.shape[1]):
        y_coords[:, i] = -(((My - 1) / 2) - i)

    phi_image = scipy.arctan2(y_coords, x_coords)
    L1 = np.abs(np.minimum(np.abs(phi_image - phi0), np.abs(phi_image + 2 * np.pi - phi0)))
    L2 = np.abs(np.minimum(np.abs(phi_image - phi0 - np.pi),
                           np.abs(phi_image + 2 * np.pi - phi0 - np.pi)))
    exp_phi = np.exp(-np.power(np.minimum(L2, L1), 2.0) / (2 * sigma * sigma)) * (1.0 / W_dist)
    effective_k = 2 * np.pi * k / np.log(np.max(W_dist))
    filter_real = exp_phi * np.cos((effective_k * (log_r)) + phase) * Mask
    filter_imag = exp_phi * np.sin((effective_k * (log_r)) + phase) * Mask

    return filter_real, filter_imag, effective_k


class steerable_conv(nn.Module):

    def __init__(self,
                 kernel_size,
                 in_channels,
                 out_channels,
                 k_range=[2],
                 phi_range=np.linspace(-np.pi, np.pi, 9),
                 sigma_phi_range=[np.pi / 8],
                 ker_size_range=np.arange(3, 15, 2),
                 phase_range=[0, np.pi / 2],
                 basis_scale=[1.0],
                 drop_rate=1.0):
        super(steerable_conv, self).__init__()

        basis_size = len(phi_range) * len(sigma_phi_range) * len(phase_range) * len(basis_scale)
        self.mult_real = Parameter(
            torch.Tensor(len(k_range), out_channels, in_channels, basis_size))
        self.mult_imag = Parameter(
            torch.Tensor(len(k_range), out_channels, in_channels, basis_size))

        self.num_scales = len(ker_size_range)
        self.scale_range = np.ones(self.num_scales)

        for i in range(self.num_scales):
            self.scale_range[i] = ker_size_range[i] / kernel_size[0]

        self.ker_size_range = ker_size_range

        max_size = self.ker_size_range[-1]

        self.filter_real = Parameter(torch.zeros(len(k_range), max_size, max_size, basis_size),
                                     requires_grad=False)
        self.filter_imag = Parameter(torch.zeros(len(k_range), max_size, max_size, basis_size),
                                     requires_grad=False)

        self.greedy_multiplier = 1
        self.k_range = k_range

        self.max_size = max_size
        self.const_real = Parameter(torch.Tensor(out_channels, in_channels))
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.basis_size = basis_size
        self.kernel_size = kernel_size
        self.effective_k = np.zeros(len(k_range))

        self.init_he = torch.zeros(len(k_range), basis_size)

        with torch.no_grad():
            for i in range(len(k_range)):
                count = 0
                for j in range(len(phi_range)):
                    for k in range(len(sigma_phi_range)):
                        for p in range(len(phase_range)):
                            for b in range(len(basis_scale)):
                                filter_real, filter_imag, eff_k = generate_filter_basis([max_size, max_size],
                                                                                        phi_range[j], sigma_phi_range[k],
                                                                                        k_range[i], basis_scale[b], phase_range[p], drop_rate)
                                filter_real = filter_real / (np.linalg.norm(filter_real))
                                filter_imag = filter_imag / (np.linalg.norm(filter_imag))
                                self.effective_k[i] = eff_k

                                self.init_he[i, count] = 2 / (
                                    basis_size * in_channels * out_channels * torch.pow(torch.norm(torch.from_numpy(filter_real)), 2.0))
                                self.filter_real[i, :, :, count] = torch.from_numpy(filter_real)
                                self.filter_imag[i, :, :, count] = torch.from_numpy(filter_imag)
                                count = count + 1

        self.reset_parameters()

    def combination(self):
        device = self.filter_real.device
        W_all = []
        Smid = int((self.max_size - 1) / 2)

        # Below: Whether to use all filter orders at all scales or not
        k_num_scales = np.ones(self.num_scales) * len(self.k_range)

        for i in range(self.num_scales):
            s = self.scale_range[i]
            Swid = int((self.ker_size_range[i] - 1) / 2)
            W_real = torch.zeros(len(self.k_range), self.out_channels, self.in_channels,
                                 self.ker_size_range[i], self.ker_size_range[i], device=device)
            W_imag = torch.zeros(len(self.k_range), self.out_channels, self.in_channels,
                                 self.ker_size_range[i], self.ker_size_range[i], device=device)

            mul = 1
            #

            for k in range(int(k_num_scales[i])):
                k_val = self.effective_k[k]
                mult_real_k = self.mult_real[k, :, :, :] * np.cos(-k_val * np.log(
                    s)) - self.mult_imag[k, :, :, :] * np.sin(-k_val * np.log(s))
                mult_imag_k = self.mult_real[k, :, :, :] * np.sin(-k_val * np.log(
                    s)) + self.mult_imag[k, :, :, :] * np.cos(-k_val * np.log(s))
                W_real[k, :, :, :, :] = torch.einsum("ijk,abk->ijab", mult_real_k,
                                                     self.filter_real[k, Smid - Swid:Smid + Swid + 1, Smid - Swid:Smid + Swid + 1, :]).contiguous()
                W_imag[k, :, :, :, :] = torch.einsum("ijk,abk->ijab", mult_imag_k,
                                                     self.filter_imag[k, Smid - Swid:Smid + Swid + 1, Smid - Swid:Smid + Swid + 1, :]).contiguous()

            W_final = torch.sum(W_real, 0) - torch.sum(W_imag, 0)
            W_all.append(W_final)

        return W_all

    def forward(self):
        return self.combination()

    def reset_parameters(self):

        # he =  0.2 / basis_size
        self.const_real.data.uniform_(-0.00001, 0.00001)

        for i in range(self.mult_real.shape[3]):
            for k in range(len(self.k_range)):
                self.mult_real[k, :, :, i].data.uniform_(-torch.sqrt(
                    self.init_he[k, i]), torch.sqrt(self.init_he[k, i]))
                self.mult_imag[k, :, :, i].data.uniform_(-torch.sqrt(
                    self.init_he[k, i]), torch.sqrt(self.init_he[k, i]))


class ScaleConv_steering(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 n_scales_small=5,
                 n_scales_big=3,
                 mode=1,
                 angle_range=120,
                 k_range=[0.5, 1, 2],
                 phi_range=np.linspace(0, np.pi, 9),
                 sigma_phi_range=[np.pi / 16],
                 ker_size_range=np.arange(3, 17, 2),
                 phase_range=[-np.pi / 4],
                 basis_scale=[1.0],
                 drop_rate=1.0,
                 relu=True):

        super(ScaleConv_steering, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ker_size_range = ker_size_range

        self.n_scales_small = n_scales_small
        self.n_scales_big = n_scales_big
        self.n_scales = n_scales_small + n_scales_big
        self.angle_range = angle_range
        self.mode = mode
        # Angles
        self.angles = np.linspace(-angle_range * self.n_scales_small / self.n_scales,
                                  angle_range * self.n_scales_big / self.n_scales, self.n_scales, endpoint=True)

        self.steer_conv = steerable_conv(self.kernel_size, in_channels, out_channels, k_range, phi_range,
                                         sigma_phi_range, ker_size_range, phase_range, basis_scale, drop_rate)

        # apply relu or not
        self.relu = relu

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)

    def _apply(self, func):
        # This is called whenever user calls model.cuda()
        # We intersect to replace tensors and variables with cuda-versions
        super(ScaleConv_steering, self)._apply(func)

    def forward(self, input):
        outputs = []
        orig_size = list(input.data.shape[2:4])

        self.weight_all = self.steer_conv()

        for i in range(len(self.weight_all)):
            padding = int((self.ker_size_range[i] - 1) / 2)
            out = F.conv2d(input, self.weight_all[i], None, self.stride, padding, self.dilation)
            outputs.append(out.unsqueeze(-1))

        strength, _ = torch.max(torch.cat(outputs, -1), -1)

        if self.relu:
            strength = F.relu(strength)
        return strength
