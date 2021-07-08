# -*- coding: utf-8
# https://github.com/mlmed/torchxrayvision/blob/master/torchxrayvision/models.py
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings


__all__ = ["DenseNet",
           "densenet121",
           "densenet161",
           "densenet169",
           "densenet201",
           "densenet121bn",
           "densenet161bn",
           "densenet169bn",
           "densenet201bn"]

warnings.filterwarnings("ignore")

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, use_gnorm: bool = True,
                 group_channels: int = 32):
        super(_DenseLayer, self).__init__()
        if use_gnorm:
            num_groups = ((num_input_features+group_channels-1)
                          // group_channels)
            self.add_module('norm1',
                            nn.GroupNorm(num_groups, num_input_features))
        else:
            self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1, bias=False)),
        if use_gnorm:
            feautres = bn_size * growth_rate
            num_groups = (feautres+group_channels-1)//group_channels
            self.add_module('norm2', nn.GroupNorm(num_groups, feautres)),
        else:
            self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        if drop_rate > 0.0:
            self.add_module("dropout", nn.Dropout2d(drop_rate))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, use_gnorm: bool = True,
                 group_channels: int = 32):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i *
                                growth_rate, growth_rate, bn_size, drop_rate,
                                use_gnorm=use_gnorm,
                                group_channels=group_channels)
            self.add_module('denselayer%d' % (i + 1), layer)

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, use_gnorm: bool = True,
                 group_channels: int = 32, use_pooling: bool = True):
        super(_Transition, self).__init__()
        if use_gnorm:
            num_groups = (num_input_features+group_channels-1)//group_channels
            self.add_module('norm',
                            nn.GroupNorm(num_groups, num_input_features))
        else:
            self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        if use_pooling:
            self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4,
                 drop_rate=0.0, last_drop_rate=0.0, num_classes=18, in_channels=1, weights=None, op_threshs=None, progress=True,
                 use_gnorm: bool = True, group_channels: int = 32):

        super(DenseNet, self).__init__()

        # First convolution
        num_groups = (num_init_features+group_channels-1)//group_channels
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels, num_init_features,
                                kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', (nn.GroupNorm(num_groups, num_init_features) if use_gnorm else
                       nn.BatchNorm2d(num_init_features))),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate,
                                drop_rate=(last_drop_rate
                                           if i == len(block_config) - 1 else drop_rate),
                                use_gnorm=use_gnorm, group_channels=group_channels)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2,
                                    use_gnorm=use_gnorm, group_channels=group_channels,
                                    use_pooling=i > 1)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        num_groups = (num_features+group_channels-1)//group_channels
        self.features.add_module('norm5', nn.GroupNorm(num_groups, num_features)
                                 if use_gnorm else nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # needs to be register_buffer here so it will go to cuda/cpu easily
        self.register_buffer('op_threshs', op_threshs)


    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = self.classifier(out)

        if hasattr(self, "op_threshs") and (self.op_threshs != None):
            out = torch.sigmoid(out)
            out = op_norm(out, self.op_threshs)

        return out


def op_norm(outputs, op_threshs):
    op_threshs = op_threshs.expand(outputs.shape[0], -1)
    outputs_new = torch.zeros(outputs.shape, device=outputs.device)
    mask_leq = outputs < op_threshs
    outputs_new[mask_leq] = outputs[mask_leq]/(op_threshs[mask_leq]*2)
    outputs_new[~mask_leq] = 1.0 - \
        ((1.0 - outputs[~mask_leq])/((1-op_threshs[~mask_leq])*2))

    return outputs_new


def get_densenet_params(arch):
    assert 'dense' in arch
    if arch == 'densenet161':
        ret = dict(growth_rate=48,
                   block_config=(6, 12, 36, 24),
                   num_init_features=96)
    elif arch == 'densenet169':
        ret = dict(growth_rate=32,
                   block_config=(6, 12, 32, 32),
                   num_init_features=64)
    elif arch == 'densenet201':
        ret = dict(growth_rate=32,
                   block_config=(6, 12, 48, 32),
                   num_init_features=64)
    else:
        # default configuration: densenet121
        ret = dict(growth_rate=32,
                   block_config=(6, 12, 24, 16),
                   num_init_features=64)
    return ret


def densenet121(bn_size: int = 4,
                drop_rate: float = 0.0,
                last_drop_rate: float = 0.0,
                num_classes: int = 18,
                in_channels: int = 1,
                weights=None,
                op_threshs=None,
                progress=True,
                group_channels: int = 32):
    params = get_densenet_params("densenet121")
    return DenseNet(bn_size=bn_size,
                    drop_rate=drop_rate,
                    last_drop_rate=last_drop_rate,
                    num_classes=num_classes,
                    in_channels=in_channels,
                    weights=weights,
                    op_threshs=op_threshs,
                    progress=progress,
                    group_channels=group_channels,
                    **params)


def densenet161(bn_size: int = 4,
                drop_rate: float = 0.0,
                last_drop_rate: float = 0.0,
                num_classes: int = 18,
                in_channels: int = 1,
                weights=None,
                op_threshs=None,
                progress=True,
                group_channels: int = 32):
    params = get_densenet_params("densenet161")
    return DenseNet(bn_size=bn_size,
                    drop_rate=drop_rate,
                    last_drop_rate=last_drop_rate,
                    num_classes=num_classes,
                    in_channels=in_channels,
                    weights=weights,
                    op_threshs=op_threshs,
                    progress=progress,
                    group_channels=group_channels,
                    **params)


def densenet169(bn_size: int = 4,
                drop_rate: float = 0.0,
                last_drop_rate: float = 0.0,
                num_classes: int = 18,
                in_channels: int = 1,
                weights=None,
                op_threshs=None,
                progress=True,
                group_channels: int = 32):
    params = get_densenet_params("densenet169")
    return DenseNet(bn_size=bn_size,
                    drop_rate=drop_rate,
                    last_drop_rate=last_drop_rate,
                    num_classes=num_classes,
                    in_channels=in_channels,
                    weights=weights,
                    op_threshs=op_threshs,
                    progress=progress,
                    group_channels=group_channels,
                    **params)


def densenet201(bn_size: int = 4,
                drop_rate: float = 0.0,
                last_drop_rate: float = 0.0,
                num_classes: int = 18,
                in_channels: int = 1,
                weights=None,
                op_threshs=None,
                progress=True,
                group_channels: int = 32):
    params = get_densenet_params("densenet201")
    return DenseNet(bn_size=bn_size,
                    drop_rate=drop_rate,
                    last_drop_rate=last_drop_rate,
                    num_classes=num_classes,
                    in_channels=in_channels,
                    weights=weights,
                    op_threshs=op_threshs,
                    progress=progress,
                    group_channels=group_channels,
                    **params)


def densenet121bn(bn_size: int = 4,
                  drop_rate: float = 0.0,
                  last_drop_rate: float = 0.0,
                  num_classes: int = 18,
                  in_channels: int = 1,
                  weights=None,
                  op_threshs=None,
                  progress=True,
                  group_channels: int = 32):
    params = get_densenet_params("densenet121")
    return DenseNet(bn_size=bn_size,
                    drop_rate=drop_rate,
                    last_drop_rate=last_drop_rate,
                    num_classes=num_classes,
                    in_channels=in_channels,
                    weights=weights,
                    op_threshs=op_threshs,
                    progress=progress,
                    use_gnorm=False,
                    group_channels=group_channels,
                    **params)


def densenet161bn(bn_size: int = 4,
                  drop_rate: float = 0.0,
                  last_drop_rate: float = 0.0,
                  num_classes: int = 18,
                  in_channels: int = 1,
                  weights=None,
                  op_threshs=None,
                  progress=True,
                  group_channels: int = 32):
    params = get_densenet_params("densenet161")
    return DenseNet(bn_size=bn_size,
                    drop_rate=drop_rate,
                    last_drop_rate=last_drop_rate,
                    num_classes=num_classes,
                    in_channels=in_channels,
                    weights=weights,
                    op_threshs=op_threshs,
                    progress=progress,
                    use_gnorm=False,
                    group_channels=group_channels,
                    **params)


def densenet169bn(bn_size: int = 4,
                  drop_rate: float = 0.0,
                  last_drop_rate: float = 0.0,
                  num_classes: int = 18,
                  in_channels: int = 1,
                  weights=None,
                  op_threshs=None,
                  progress=True,
                  group_channels: int = 32):
    params = get_densenet_params("densenet169")
    return DenseNet(bn_size=bn_size,
                    drop_rate=drop_rate,
                    last_drop_rate=last_drop_rate,
                    num_classes=num_classes,
                    in_channels=in_channels,
                    weights=weights,
                    op_threshs=op_threshs,
                    progress=progress,
                    use_gnorm=False,
                    group_channels=group_channels,
                    **params)


def densenet201bn(bn_size: int = 4,
                  drop_rate: float = 0.0,
                  last_drop_rate: float = 0.0,
                  num_classes: int = 18,
                  in_channels: int = 1,
                  weights=None,
                  op_threshs=None,
                  progress=True,
                  group_channels: int = 32):
    params = get_densenet_params("densenet201")
    return DenseNet(bn_size=bn_size,
                    drop_rate=drop_rate,
                    last_drop_rate=last_drop_rate,
                    num_classes=num_classes,
                    in_channels=in_channels,
                    weights=weights,
                    op_threshs=op_threshs,
                    progress=progress,
                    use_gnorm=False,
                    group_channels=group_channels,
                    **params)
