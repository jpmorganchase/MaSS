# SPDX-License-Identifier: Apache-2.0
# Copyright : JP Morgan Chase & Co
 
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

from timm.models import register_model

class MLP(nn.Module):
    def __init__(
        self,
        widths,
        bn=True,
    ):
        super().__init__()
        self.bn = bn

        layers = []
        for i in np.arange(len(widths) - 1):
            fc_layer = nn.Linear(widths[i], widths[i + 1])
            layers.append(fc_layer)

            if i != len(widths) - 2:
                if bn:
                    layers.append(nn.BatchNorm1d(widths[i + 1]))

                layers.append(nn.ReLU(True))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        return x


class ResMLP(nn.Module):
    def __init__(self, widths, bn=True):
        super(ResMLP, self).__init__()

        input_dim = widths[0]
        output_dim = widths[-1]

        self.downsample = None
        if input_dim != output_dim:
            self.downsample = nn.Linear(input_dim, output_dim)
        self.mlp = MLP(widths, bn=bn)
        self.relu = nn.ReLU(True)
        if bn:
            self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        identity = x
        out = self.mlp(x)
        out = self.bn(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class ResMLPGeneratorWUnitGaussianNoise(nn.Module):

    def __init__(self, layer_dims: List[int], num_block=1, noise_dim: int = 0):
        super().__init__()

        layer_dims1 = copy.deepcopy(layer_dims)
        layer_dims1[0] += noise_dim
        self.layer1 = MLP(layer_dims1)
        self.bn1 = nn.BatchNorm1d(layer_dims1[-1])
        self.relu = nn.ReLU()

        layers = []
        for i in range(num_block-1):
            layers.append(ResMLP(layer_dims, True))
        layers.append(nn.Linear(layer_dims[0], layer_dims[-1]))
        self.layers = nn.Sequential(*layers)

        self.noise_dim = noise_dim
        # weight initialization

    def forward(self, x):
        x = x.reshape(x.shape[0], 256)
        noise = torch.randn((x.shape[0], self.noise_dim), device=x.device)
        out = torch.cat([x, noise], dim=1)
        out = self.layer1(out)
        out = self.bn1(out)
        out = out + x
        out = self.relu(out)
        out = self.layers(out)
        out = out.reshape(out.shape[0], 2, 128)

        return out


@register_model
def motionsense_gen_mlp(in_chans, **kwargs):

    layer_cfg = [256, 256, 256]
    model = ResMLPGeneratorWUnitGaussianNoise(
        layer_dims=layer_cfg, num_block=1, noise_dim=256)
    return model