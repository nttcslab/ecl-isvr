# -*- coding: utf-8
import torch

__all__ = ["LogisticsModel"]


class LogisticsModel(torch.nn.Module):
    def __init__(self, in_features: int = 784, num_classes: int = 26):
        super(LogisticsModel, self).__init__()
        # EMNIST -> Image size: 1x28x28(=784), class: 26 + 1
        # CIFAR-10 -> Image size: 3x32x32(=3072), class: 10
        self.linear = torch.nn.Linear(in_features, num_classes)

    def forward(self, x):
        if x.dim() > 2:
            x = x.flatten(start_dim=1)
        return self.linear(x)
