from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.registry import register_model


class MotionSenseMLP(nn.Module):

    def __init__(self, in_chans, num_classes):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv1d(in_chans, 16, 3, 1, 1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 3, 1, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, 1, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, 3, 1, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

        # weight initialization

    def forward(self, x, feature=False):
        if feature:
            return self.layers(x), None
        else:
            return self.layers(x)


@register_model
def motionsense_mlp(in_chans, num_classes, **kwargs):
    model = MotionSenseMLP(in_chans=in_chans, num_classes=num_classes)
    return model

