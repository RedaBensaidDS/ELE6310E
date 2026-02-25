"""
Modified from https://github.com/chenyaofo/pytorch-cifar-models
"""
import sys
import torch.nn as nn
import torch
from common.utils import Quantized_Linear, Quantized_Conv2d

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from functools import partial
from typing import Dict, Type, Any, Callable, Union, List, Optional

cifar10_pretrained_weight_url = 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/resnet/cifar10_resnet32-ef93fc4d.pt'


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return Quantized_Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return Quantized_Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

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


class CifarResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(CifarResNet, self).__init__()
        self.inplanes = 16
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = Quantized_Linear(64 * block.expansion, num_classes, bias=True)

        for m in self.modules():
            if isinstance(m, Quantized_Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def _resnet(
        arch: str,
        layers: List[int],
        model_url: str,
        save_path: str = './save/',
        progress: bool = True,
        pretrained: bool = False,
        device=torch.device('cpu'),
        **kwargs: Any
) -> CifarResNet:
    model = CifarResNet(BasicBlock, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_url, save_path,
                                              map_location=device, progress=progress)
        network_kvpair = model.state_dict()
        for key in state_dict.keys():
            network_kvpair[key] = state_dict[key]
        model.load_state_dict(network_kvpair)
    return model


def resnet32(pretrained=False, progress=True, device=torch.device('cpu'), **kwargs):
    """Constructs a ResNet-32 model for CIFAR-10 (by default)
    Args:
      device:
      progress:
      pretrained:
    """
    save_path = kwargs.pop('save_path')
    return _resnet("resnet32", [5, 5, 5], cifar10_pretrained_weight_url, save_path, progress, pretrained, device,
                   **kwargs)


def resnet_from_scratch(
    *,
    # choose depth OR layers (exactly one)
    depth: Optional[int] = None,            # CIFAR ResNet depth = 6*n + 2 (e.g., 20, 32, 44, 56, 110)
    layers: Optional[List[int]] = None,     # explicit blocks per stage: [l1, l2, l3]

    # "feature maps per layer" (per stage in this CIFAR ResNet): [c1, c2, c3]
    channels: List[int] = [16, 32, 64],

    # usual args
    num_classes: int = 10,
    in_channels: int = 3,                   # in case you want grayscale etc.
    **kwargs: Any
) -> CifarResNet:
    """
    Builds a CIFAR-style ResNet from scratch with customizable:
      - number of layers: via `depth` or `layers`
      - feature maps: via `channels` = [c1, c2, c3] for stages 1/2/3

    Notes:
      - This uses the existing CifarResNet/BasicBlock code, so the network
        has 3 stages. "channels" controls the width of each stage.
      - The original CifarResNet class hardcodes width to [16,32,64].
        To keep your original functions unchanged, this function defines
        a small subclass that reuses the same logic but swaps in `channels`.
    """
    if (depth is None) == (layers is None):
        raise ValueError("Specify exactly one of `depth` or `layers`.")

    if layers is None:
        if (depth - 2) % 6 != 0:
            raise ValueError(f"Invalid depth={depth}. For CIFAR BasicBlock ResNet, depth must be 6*n + 2.")
        n = (depth - 2) // 6
        if n <= 0:
            raise ValueError(f"Invalid depth={depth}. Must be >= 8 (n>=1).")
        layers = [n, n, n]

    if len(layers) != 3:
        raise ValueError(f"`layers` must have length 3, got {layers}")
    if len(channels) != 3:
        raise ValueError(f"`channels` must have length 3, got {channels}")

    c1, c2, c3 = channels

    class _CifarResNetCustomWidths(CifarResNet):
        def __init__(self, block, layers, num_classes=10):
            super(CifarResNet, self).__init__()  # bypass CifarResNet init, keep rest of file intact
            self.inplanes = c1

            self.conv1 = conv3x3(in_channels, c1)
            self.bn1 = nn.BatchNorm2d(c1)
            self.relu = nn.ReLU(inplace=True)

            self.layer1 = self._make_layer(block, c1, layers[0])
            self.layer2 = self._make_layer(block, c2, layers[1], stride=2)
            self.layer3 = self._make_layer(block, c3, layers[2], stride=2)

            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = Quantized_Linear(c3 * block.expansion, num_classes, bias=True)

            for m in self.modules():
                if isinstance(m, Quantized_Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    # Build from scratch (no pretrained loading)
    return _CifarResNetCustomWidths(BasicBlock, layers, num_classes=num_classes, **kwargs)