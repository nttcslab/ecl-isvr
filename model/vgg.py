#! -*- coding: utf-8
# cloned from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
from collections import OrderedDict

import torch
import torch.nn as nn

__all__ = ["VGG",
           "vgg11", "vgg11_bn",
           "vgg13", "vgg13_bn",
           "vgg16", "vgg16_bn",
           "vgg19", "vgg19_bn",
           "vgg11s", "vgg11s_bn",
           "vgg13s", "vgg13s_bn",
           "vgg16s", "vgg16s_bn",
           "vgg19s", "vgg19s_bn"]


class VGG(nn.Module):

    def __init__(self, features,
                 num_classes=1000,
                 out_channels: int = 512,
                 linear_features: list = [4096, 4096],
                 init_weights=True,
                 drop_rate: float = 0.0):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        classifier = OrderedDict()
        in_fature = out_channels * 7 * 7
        out_features = linear_features + [num_classes]
        for i, out_feature in enumerate(out_features, 1):
            classifier["linear%d" % i] = nn.Linear(in_fature, out_feature)
            if i < len(out_features):
                classifier["relu%d" % i] = nn.ReLU(True)
                if drop_rate > 0.0:
                    classifier["dropout%d" % i] = nn.Dropout(p=drop_rate)
            in_fature = out_feature
        self.classifier = nn.Sequential(classifier)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, in_channels: int = 3, batch_norm=False, drop_rate: float = 0.0, group_channels: int = 32):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                # layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                # num_groups = (v + in_channels - 1) // v
                num_groups = (in_channels+group_channels-1)//group_channels
                layers += [conv2d,
                           nn.GroupNorm(num_groups, v),
                           nn.ReLU(inplace=True)]
            elif drop_rate > 0.0:
                layers += [conv2d,
                           nn.ReLU(inplace=True),
                           nn.Dropout2d(p=drop_rate)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'a': [32, 'M', 64, 'M', 128, 128, 'M', 256, 256, 'M', 256, 256, 'M'],
    'b': [32, 32, 'M', 64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 256, 256, 'M'],
    'd': [32, 32, 'M', 64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M'],
    'e': [32, 32, 'M', 64, 64, 'M', 128, 128, 128, 128, 'M', 256, 256, 256, 256, 'M', 256, 256, 256, 256, 'M'],
    "a1": [16, 'M', 16, 'M', 16, 'M', 16, 'M', 16, 'M'],
    "a2": [32, 'M', 64, 'M', 128, 'M', 128, 'M'],
}


def _vgg(arch, cfg, batch_norm,
         linear_features: int = [4096, 4096],
         in_channels: int = 3,
         drop_rate: float = 0.0,
         last_drop_rate: float = 0.0,
         group_channels: int = 32,
         **kwargs):
    config = cfgs[cfg]
    model = VGG(make_layers(config,
                            in_channels=in_channels,
                            batch_norm=batch_norm,
                            drop_rate=drop_rate,
                            group_channels=group_channels),
                linear_features=linear_features,
                out_channels=config[-2],
                drop_rate=last_drop_rate, **kwargs)
    return model


def vgg11(**kwargs):
    return _vgg('vgg11', 'A', False, **kwargs)


def vgg11_bn(**kwargs):
    return _vgg('vgg11_bn', 'A', True, **kwargs)


def vgg13(**kwargs):
    return _vgg('vgg13', 'B', False, **kwargs)


def vgg13_bn(**kwargs):
    return _vgg('vgg13_bn', 'B', True, **kwargs)


def vgg16(**kwargs):
    return _vgg('vgg16', 'D', False, **kwargs)


def vgg16_bn(**kwargs):
    return _vgg('vgg16_bn', 'D', True, **kwargs)


def vgg19(**kwargs):
    return _vgg('vgg19', 'E', False, **kwargs)


def vgg19_bn(**kwargs):
    return _vgg('vgg19_bn', 'E', True, **kwargs)


def vgg11s(**kwargs):
    return _vgg('vgg11s', "a", False, linear_features=[1024, 1024], **kwargs)


def vgg11s_bn(**kwargs):
    return _vgg('vgg11s_bn', "a", True, linear_features=[1024, 1024], **kwargs)


def vgg13s(**kwargs):
    return _vgg('vgg13s', "b", False, linear_features=[1024, 1024], **kwargs)


def vgg13s_bn(**kwargs):
    return _vgg('vgg13s_bn', "b", True, linear_features=[1024, 1024], **kwargs)


def vgg16s(**kwargs):
    return _vgg('vgg16s', "d", False, linear_features=[1024, 1024], **kwargs)


def vgg16s_bn(**kwargs):
    return _vgg('vgg16s_bn', "d", True, linear_features=[1024, 1024], **kwargs)


def vgg19s(**kwargs):
    return _vgg('vgg19s', "e", False, linear_features=[1024, 1024], **kwargs)


def vgg19s_bn(**kwargs):
    return _vgg('vgg19s_bn', "e", True, linear_features=[1024, 1024], **kwargs)
