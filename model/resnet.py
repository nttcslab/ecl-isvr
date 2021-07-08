#! -*- coding: utf-8
import torch
import torch.nn as nn

__all__ = ["BasicBlock", "Bottleneck", "conv3x3",
           "conv1x1", "ResNet",
           "resnet10", "resnet18", "resnet32",
           "resnet50", "resnet101", "resnet152"]


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, group_channels: int = 32):
        super(BasicBlock, self).__init__()
        self._group_channels = group_channels
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        if norm_layer == nn.GroupNorm:
            # num_groups = (planes+inplanes-1)//inplanes
            num_groups = (planes+self.group_channels-1)//self.group_channels
            self.bn1 = norm_layer(num_groups, planes)
        else:
            self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        if norm_layer == nn.GroupNorm:
            # num_groups = (planes+planes-1)//planes
            num_groups = (planes+self.group_channels-1)//self.group_channels
            self.bn2 = norm_layer(num_groups, planes)
        else:
            self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    @property
    def group_channels(self) -> int: return self._group_channels

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, group_channels: int = 32):
        super(Bottleneck, self).__init__()
        self._group_channels = group_channels
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        if norm_layer == nn.GroupNorm:
            # num_groups = (width+inplanes-1)//inplanes
            num_groups = (width+self.group_channels-1)//self.group_channels
            self.bn1 = norm_layer(num_groups, width)
        else:
            self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        if norm_layer == nn.GroupNorm:
            # num_groups = (width+width-1)//width
            num_groups = (width+self.group_channels-1)//self.group_channels
            self.bn2 = norm_layer(num_groups, width)
        else:
            self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        if norm_layer == nn.GroupNorm:
            # num_groups = (planes*self.expansion+width-1)//width
            num_groups = (planes*self.expansion +
                          self.group_channels-1)//self.group_channels
            self.bn3 = norm_layer(num_groups, planes * self.expansion)
        else:
            self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    @property
    def group_channels(self) -> int: return self._group_channels

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers,
                 in_channel: int = 3,
                 num_classes=1000,
                 zero_init_residual=False,
                 groups=1,
                 width_per_group=64,
                 replace_stride_with_dilation=None,
                 norm_layer=None,
                 drop_rate: float = 0.0,
                 last_drop_rate: float = 0.0,
                 group_channels: int = 32):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self._drop_rate = drop_rate
        self._group_channels = group_channels

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(in_channel, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        if norm_layer == nn.GroupNorm:
            # num_groups = (self.inplanes + in_channel - 1)//in_channel
            num_groups = ((self.inplanes + self.group_channels - 1)
                          // self.group_channels)
            self.bn1 = norm_layer(num_groups, self.inplanes)
        else:
            self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        if drop_rate > 0.0:
            self.dropout1 = nn.Dropout2d(p=drop_rate)
        planes = self.inplanes
        if len(layers) >= 1:
            planes = 64
            self.layer1 = self._make_layer(block, planes, layers[0],
                                           drop_rate=0.0 if len(layers) == 1 else drop_rate)
        if len(layers) >= 2:
            planes = 128
            self.layer2 = self._make_layer(block, planes, layers[1], stride=2,
                                           dilate=replace_stride_with_dilation[0],
                                           drop_rate=0.0 if len(layers) == 2 else drop_rate)
        if len(layers) >= 3:
            planes = 256
            self.layer3 = self._make_layer(block, planes, layers[2], stride=2,
                                           dilate=replace_stride_with_dilation[1],
                                           drop_rate=0.0 if len(layers) == 3 else drop_rate)
        if len(layers) >= 4:
            planes = 512
            self.layer4 = self._make_layer(block, planes, layers[3], stride=2,
                                           dilate=replace_stride_with_dilation[2],
                                           drop_rate=0.0)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if last_drop_rate > 0.0:
            self.dropout_fc = nn.Dropout(p=last_drop_rate)

        self.fc = nn.Linear(planes * block.expansion, num_classes)
        # self.fc_score = nn.Linear(planes * block.expansion, num_classes)
        # if num_classes == 1:
        #     self.fc_prob = nn.Sequential(nn.Linear(num_classes, num_classes),
        #                                  nn.Sigmoid())
        # else:
        #     self.fc_prob = nn.Sequential(nn.Linear(num_classes, num_classes),
        #                                  nn.Softmax())

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(
        #             m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

        # # Zero-initialize the last BN in each residual branch,
        # # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # if zero_init_residual:
        #     for m in self.modules():
        #         if isinstance(m, Bottleneck):
        #             nn.init.constant_(m.bn3.weight, 0)
        #         elif isinstance(m, BasicBlock):
        #             nn.init.constant_(m.bn2.weight, 0)
    @property
    def group_channels(self) -> int: return self._group_channels

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, drop_rate: float = 0.0):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if norm_layer == nn.GroupNorm:
                # num_groups = (planes * block.expansion +
                #               self.inplanes - 1) // self.inplanes
                out_channels = planes * block.expansion
                num_groups = ((out_channels + self.group_channels - 1)
                              // self.group_channels)
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, out_channels, stride),
                    norm_layer(num_groups, out_channels),)
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),)

        layers = []
        layers.append(block(self.inplanes, planes, stride,
                            downsample=downsample, groups=self.groups,
                            base_width=self.base_width, dilation=previous_dilation,
                            norm_layer=norm_layer, group_channels=self.group_channels))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer,
                                group_channels=self.group_channels))

        if drop_rate > 0.0:
            layers.append(nn.Dropout2d(p=drop_rate))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        if hasattr(self, "dropout1"):
            x = self.dropout1(x)

        if hasattr(self, "layer1"):
            x = self.layer1(x)
        if hasattr(self, "layer2"):
            x = self.layer2(x)
        if hasattr(self, "layer3"):
            x = self.layer3(x)
        if hasattr(self, "layer4"):
            x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if hasattr(self, "dropout_fc"):
            x = self.dropout_fc(x)
        x = self.fc(x)
        return x

        # x = self.fc_score(x)

        # if hasattr(self, "fc_prob"):
        #     x_prob = self.fc_prob(x)
        # else:
        #     x_prob = None

        # return x, x_prob

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(block, layers, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


def resnet10(in_channel: int = 3,
             drop_rate: float = 0.0,
             last_drop_rate: float = 0.0,
             **kwargs):
    return _resnet(BasicBlock, [1, 1, 1, 1],
                   in_channel=in_channel,
                   drop_rate=drop_rate,
                   last_drop_rate=last_drop_rate,
                   **kwargs)


def resnet18(in_channel: int = 3,
             drop_rate: float = 0.0,
             last_drop_rate: float = 0.0,
             **kwargs):
    return _resnet(BasicBlock, [2, 2, 2, 2],
                   in_channel=in_channel,
                   drop_rate=drop_rate,
                   last_drop_rate=last_drop_rate,
                   **kwargs)


def resnet32(in_channel: int = 3,
             drop_rate: float = 0.0,
             last_drop_rate: float = 0.0,
             **kwargs):
    return _resnet(BasicBlock, [3, 4, 6, 3],
                   in_channel=in_channel,
                   drop_rate=drop_rate,
                   last_drop_rate=last_drop_rate,
                   **kwargs)


def resnet50(in_channel: int = 3,
             drop_rate: float = 0.0,
             last_drop_rate: float = 0.0,
             **kwargs):
    return _resnet(Bottleneck, [3, 4, 6, 3],
                   in_channel=in_channel,
                   drop_rate=drop_rate,
                   last_drop_rate=last_drop_rate,
                   **kwargs)


def resnet101(in_channel: int = 3,
              drop_rate: float = 0.0,
              last_drop_rate: float = 0.0,
              **kwargs):
    return _resnet(Bottleneck, [3, 4, 23, 3],
                   in_channel=in_channel,
                   drop_rate=drop_rate,
                   last_drop_rate=last_drop_rate,
                   **kwargs)


def resnet152(in_channel: int = 3,
              drop_rate: float = 0.0,
              last_drop_rate: float = 0.0,
              **kwargs):
    return _resnet(Bottleneck, [3, 8, 36, 3],
                   in_channel=in_channel,
                   drop_rate=drop_rate,
                   last_drop_rate=last_drop_rate,
                   **kwargs)
