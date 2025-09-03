import torch
import torch.nn as nn
from copy import deepcopy
from spikingjelly.activation_based import layer, functional
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode, MultiStepLIFNode
import torch.nn.functional as F

try:
    from torchvision.models.utils import load_state_dict_from_url
except ImportError:
    from torchvision._internally_replaced_utils import load_state_dict_from_url

__all__ = ['spikingResNet', 'spiking_resnet18', 'spiking_resnet34', 'spiking_resnet50', 'spiking_resnet101',
           'spiking_resnet152', 'spiking_resnext50_32x4d', 'spiking_resnext101_32x8d',
           'spiking_wide_resnet50_2', 'spiking_wide_resnet101_2']

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-63fe2227.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-394f9c45.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
}

import torch
import torch.nn as nn

def conv3x3_ann(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock_ann(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3_ann(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3_ann(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

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

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def resnet18(num_classes=1000):
    return ResNet(BasicBlock_ann, [2, 2, 2, 2], num_classes)


def spiking_function(x: torch.Tensor, y: torch.Tensor, cnf: str):
    if cnf == 'ADD':
        return x + y
    elif cnf == 'AND':
        return x * y
    elif cnf == 'IAND':
        return x * (1. - y)
    else:
        raise NotImplementedError


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return layer.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                        padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return layer.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, cnf: str = None,
                 **kwargs):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.sn1 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        # self.sn2 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.downsample = downsample
        # if downsample is not None:
        #     self.downsample_sn = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.stride = stride
        self.cnf = cnf

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.sn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.sn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = spiking_function(identity, out, self.cnf)    
        return out
    def extra_repr(self) -> str:
        return super().extra_repr() + f'cnf={self.cnf}'


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, cnf: str = None,
                 **kwargs):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.sn1 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.sn2 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.sn3 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.downsample = downsample
        # if downsample is not None:
        #     self.downsample_sn = mem_update()
        self.stride = stride
        self.cnf = cnf

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.sn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.sn2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.sn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # out = spiking_function(out, identity, self.cnf)
        out = out + identity

        return out

    def extra_repr(self) -> str:
        return super().extra_repr() + f'cnf={self.cnf}'


class spikingResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, cnf: str = None, pe: bool=False, **kwargs):
        super().__init__()
        self.T = kwargs['T']
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        self._norm_layer = norm_layer

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
        self.conv1 = layer.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                                  bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.maxpool = layer.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.sn1 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.layer1 = self._make_layer(block, 64, layers[0], cnf=cnf, **kwargs)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], cnf=cnf, **kwargs)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], cnf=cnf, **kwargs)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], cnf=cnf, **kwargs)
        

        self.layer1_sn_1 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.layer2_sn_1 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.layer3_sn_1 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.layer4_sn_1 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')


        self.layer1_sn = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.layer2_sn = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.layer3_sn = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.layer4_sn = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.avgpool = layer.AdaptiveAvgPool2d((1, 1))
        self.fc = layer.Linear(512 * block.expansion, num_classes)

        self.inplanes = 64  
        self.conv1_ann = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_ann = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool_ann = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.annlayer1 = self._make_layer_ann(BasicBlock_ann, 64, layers[0])
        self.annlayer2 = self._make_layer_ann(BasicBlock_ann, 128, layers[1], stride=2)
        self.annlayer3 = self._make_layer_ann(BasicBlock_ann, 256, layers[2], stride=2)
        # self.annlayer4 = self._make_layer_ann(BasicBlock_ann, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (layer.BatchNorm2d, layer.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        functional.set_step_mode(self, 'm')

        if pe is None:
            self.conv1.step_mode = 's'
            self.bn1.step_mode = 's'
            self.pe = None
        else:
            self.pe = nn.Parameter(torch.zeros([self.T, 1, 1, 1, 1]))
            torch.nn.init.trunc_normal_(self.pe, std=0.02)



    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, cnf: str = None,
                     **kwargs):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, cnf, **kwargs))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, cnf=cnf, **kwargs))

        return nn.Sequential(*layers)

    def _make_layer_ann(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)


    def _forward_impl(self, x):
        # See note [TorchScript super()]
        out_ann = self.conv1_ann(x)
        out_ann = self.bn1_ann(out_ann)
        out_ann = self.relu(out_ann)
        out_ann = self.maxpool_ann(out_ann)

        out_ann_layer1 = self.annlayer1(out_ann)
        annx1_add = out_ann_layer1.unsqueeze(0)
        annx1_add = annx1_add.repeat(self.T, 1, 1, 1, 1)

        out_ann_layer2 = self.annlayer2(out_ann_layer1)
        annx2_add = out_ann_layer2.unsqueeze(0)
        annx2_add = annx2_add.repeat(self.T, 1, 1, 1, 1)

        out_ann_layer3 = self.annlayer3(out_ann_layer2)
        annx3_add = out_ann_layer3.unsqueeze(0)
        annx3_add = annx3_add.repeat(self.T, 1, 1, 1, 1)


        if self.pe is None:
            out = self.conv1(x)
            out = self.bn1(out)
            out = out.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        else:
            out = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1) + self.pe
            out = self.conv1(out)
            out = self.bn1(out)
        out = self.sn1(out)
 
        out = self.maxpool(out)
        for i, block in enumerate(self.layer1):
            if i == 0:
                out = block(out)
                out = self.layer1_sn_1(out)
            if i == 1:
                out = block(out)
                out = out + annx1_add
                out = self.layer1_sn(out)
        for i, block in enumerate(self.layer2):
            if i == 0:
                out = block(out)
                out = self.layer2_sn_1(out)
            if i == 1:
                out = block(out)
                out = out + annx2_add
                out = self.layer2_sn(out)
        for i, block in enumerate(self.layer3):
            if i == 0:
                out = block(out)
                out = self.layer3_sn_1(out)
            if i == 1:
                out = block(out)
                out = out + annx3_add
                out = self.layer3_sn(out)
        for i, block in enumerate(self.layer4):
            if i == 0:
                out = block(out)
                out = self.layer4_sn_1(out)
            if i == 1:
                out = block(out)
                out = self.layer4_sn(out)


        out = self.avgpool(out)
        if self.avgpool.step_mode == 's':
            out = torch.flatten(out, 1)
        elif self.avgpool.step_mode == 'm':
            out = torch.flatten(out, 2)

        out = self.fc(out)

        return out
   


    def forward(self, x):
        return self._forward_impl(x)


def _spiking_resnet(arch, block, layers, pretrained, progress, cnf, **kwargs):
    model = spikingResNet(block, layers, cnf=cnf, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def spiking_resnet18(pretrained=False, progress=True, cnf: str = None,**kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param cnf: the name of spike-element-wise function
    :type cnf: str
    :param spiking_neuron: a spiking neuron layer
    :type spiking_neuron: callable
    :param kwargs: kwargs for `spiking_neuron`
    :type kwargs: dict
    :return: Spiking ResNet-18
    :rtype: torch.nn.Module

    The spike-element-wise ResNet-18 `"Deep Residual Learning in Spiking Neural Networks" <https://arxiv.org/abs/2102.04159>`_ modified by the ResNet-18 model from `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """

    return _spiking_resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, cnf, **kwargs)


def spiking_resnet34(pretrained=False, progress=True, cnf: str = None, **kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param cnf: the name of spike-element-wise function
    :type cnf: str
    :param spiking_neuron: a spiking neuron layer
    :type spiking_neuron: callable
    :param kwargs: kwargs for `spiking_neuron`
    :type kwargs: dict
    :return: Spiking ResNet-34
    :rtype: torch.nn.Module

    The spike-element-wise ResNet-34 `"Deep Residual Learning in Spiking Neural Networks" <https://arxiv.org/abs/2102.04159>`_
    modified by the ResNet-34 model from `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _spiking_resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress, cnf, **kwargs)


def spiking_resnet50(pretrained=False, progress=True, cnf: str = None, **kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param cnf: the name of spike-element-wise function
    :type cnf: str
    :param spiking_neuron: a spiking neuron layer
    :type spiking_neuron: callable
    :param kwargs: kwargs for `spiking_neuron`
    :type kwargs: dict
    :return: Spiking ResNet-50
    :rtype: torch.nn.Module

    The spike-element-wise ResNet-50 `"Deep Residual Learning in Spiking Neural Networks" <https://arxiv.org/abs/2102.04159>`_
    modified by the ResNet-50 model from `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _spiking_resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, cnf, **kwargs)


def spiking_resnet101(pretrained=False, progress=True, cnf: str = None, **kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param cnf: the name of spike-element-wise function
    :type cnf: str
    :param spiking_neuron: a spiking neuron layer
    :type spiking_neuron: callable
    :param kwargs: kwargs for `spiking_neuron`
    :type kwargs: dict
    :return: Spiking ResNet-101
    :rtype: torch.nn.Module

    The spike-element-wise ResNet-101 `"Deep Residual Learning in Spiking Neural Networks" <https://arxiv.org/abs/2102.04159>`_
    modified by the ResNet-101 model from `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _spiking_resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress, cnf, **kwargs)


def spiking_resnet152(pretrained=False, progress=True, cnf: str = None, **kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param cnf: the name of spike-element-wise function
    :type cnf: str
    :param spiking_neuron: a single step neuron
    :type spiking_neuron: callable
    :param kwargs: kwargs for `spiking_neuron`
    :type kwargs: dict
    :return: Spiking ResNet-152
    :rtype: torch.nn.Module

    The spike-element-wise ResNet-152 `"Deep Residual Learning in Spiking Neural Networks" <https://arxiv.org/abs/2102.04159>`_
    modified by the ResNet-152 model from `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _spiking_resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress, cnf, **kwargs)


def spiking_resnext50_32x4d(pretrained=False, progress=True, cnf: str = None,**kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param cnf: the name of spike-element-wise function
    :type cnf: str
    :param spiking_neuron: a single step neuron
    :type spiking_neuron: callable
    :param kwargs: kwargs for `spiking_neuron`
    :type kwargs: dict
    :return: Spiking ResNeXt-50 32x4d
    :rtype: torch.nn.Module

    The spike-element-wise ResNeXt-50 32x4d `"Deep Residual Learning in Spiking Neural Networks" <https://arxiv.org/abs/2102.04159>`_
    modified by the ResNeXt-50 32x4d model from `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _spiking_resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3], pretrained, progress, cnf, **kwargs)


def spiking_resnext101_32x8d(pretrained=False, progress=True, cnf: str = None, **kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param cnf: the name of spike-element-wise function
    :type cnf: str
    :param spiking_neuron: a single step neuron
    :type spiking_neuron: callable
    :param kwargs: kwargs for `spiking_neuron`
    :type kwargs: dict
    :return: Spiking ResNeXt-101 32x8d
    :rtype: torch.nn.Module

    The spike-element-wise ResNeXt-101 32x8d `"Deep Residual Learning in Spiking Neural Networks" <https://arxiv.org/abs/2102.04159>`_ modified by the ResNeXt-101 32x8d model from `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _spiking_resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3], pretrained, progress, cnf, **kwargs)



def spiking_wide_resnet50_2(pretrained=False, progress=True, cnf: str = None, **kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param cnf: the name of spike-element-wise function
    :type cnf: str
    :param spiking_neuron: a single step neuron
    :type spiking_neuron: callable
    :param kwargs: kwargs for `spiking_neuron`
    :type kwargs: dict
    :return: Spiking Wide ResNet-50-2
    :rtype: torch.nn.Module

    The spike-element-wise Wide ResNet-50-2 `"Deep Residual Learning in Spiking Neural Networks" <https://arxiv.org/abs/2102.04159>`_
    modified by the Wide ResNet-50-2 model from `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    """
    kwargs['width_per_group'] = 64 * 2
    return _spiking_resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3], pretrained, progress, cnf, **kwargs)


def spiking_wide_resnet101_2(pretrained=False, progress=True, cnf: str = None,  **kwargs):
    """
    :param pretrained: If True, the SNN will load parameters from the ANN pre-trained on ImageNet
    :type pretrained: bool
    :param progress: If True, displays a progress bar of the download to stderr
    :type progress: bool
    :param cnf: the name of spike-element-wise function
    :type cnf: str
    :param spiking_neuron: a single step neuron
    :type spiking_neuron: callable
    :param kwargs: kwargs for `spiking_neuron`
    :type kwargs: dict
    :return: Spiking Wide ResNet-101-2
    :rtype: torch.nn.Module

    The spike-element-wise Wide ResNet-101-2 `"Deep Residual Learning in Spiking Neural Networks" <https://arxiv.org/abs/2102.04159>`_
    modified by the Wide ResNet-101-2 model from `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    """
    kwargs['width_per_group'] = 64 * 2
    return _spiking_resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3], pretrained, progress, cnf,
                       **kwargs)
