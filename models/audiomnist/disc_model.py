# SPDX-License-Identifier: Apache-2.0
# Copyright : JP Morgan Chase & Co
 
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.registry import register_model


class StandardMLP(nn.Module):

    def __init__(self, layer_dims: List[int]):
        super().__init__()

        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(
                nn.Linear(layer_dims[i], layer_dims[i + 1], bias=True))
            if i != len(layer_dims) - 2:  # no activation function at the end
                layers.append(nn.BatchNorm1d(layer_dims[i + 1]))
                layers.append(nn.ReLU())

        self.layers = nn.ModuleList(layers)

    def forward(self, x, feature=False):
        features = self.extract_features(x)
        out = self.layers[-1](features)

        if feature:
            return out, features
        else:
            return out

    def extract_features(self, x):
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
        return x

class InfoNCEMLP(nn.Module):

    def __init__(self, in_chans: int, num_classes: int):
        super().__init__()
        dim_in = 512
        feat_dim = num_classes
        self.encoder = nn.Sequential(
            nn.Linear(in_chans, dim_in),
            nn.BatchNorm1d(dim_in),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in, dim_in),
            nn.BatchNorm1d(dim_in),
            nn.ReLU(inplace=True)
        )
        self.head = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in, feat_dim)
        )
        
    def forward(self, x, feature=False):
        features = self.extract_features(x)
        out = self.head(features)
        if feature:
            return out, features
        else:
            return out

    def extract_features(self, x):
        x = self.encoder(x)
        return x

@register_model
def audiomnist_mlp(in_chans, num_classes, **kwargs):
    layer_cfg = [in_chans, 512, 256, num_classes]
    model = StandardMLP(layer_dims=layer_cfg)
    return model

@register_model
def audiomnist_infonce_mlp(in_chans, num_classes, **kwargs):
    model = InfoNCEMLP(in_chans=in_chans, num_classes=num_classes)
    return model
