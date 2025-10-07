import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class GSA(nn.Module):
    def __init__(self, gate_channels):
        super(GSA, self).__init__()

        # convolution groups corresponding to different receptive sizes
        self.gate_channels = gate_channels
        self.gate_channels_1 = self.gate_channels // 2
        self.gate_channels_3 = self.gate_channels - self.gate_channels // 2

        # RF:3
        self.conv1 = nn.Conv2d(self.gate_channels_1, self.gate_channels_1, kernel_size=3, stride=1,
                               padding=(3 - 1) // 2, bias=False, groups=self.gate_channels_1)
        self.bn1 = nn.BatchNorm2d(self.gate_channels_1)

        # RF:7
        self.conv2 = nn.Conv2d(self.gate_channels_3, self.gate_channels_3, kernel_size=7, stride=1,
                               padding=(7 - 1) // 2, bias=False, groups=self.gate_channels_3)
        self.bn2 = nn.BatchNorm2d(self.gate_channels_3)

        self.relu = nn.ReLU(inplace=True)

        # adaption of channel significance
        self.fc1 = nn.Conv2d(self.gate_channels_1, self.gate_channels_1, kernel_size=(2, 1), stride=1, padding=0)
        self.fc2 = nn.Conv2d(self.gate_channels_3, self.gate_channels_3, kernel_size=(2, 1), stride=1, padding=0)

        self.sigmoid = nn.Sigmoid()

    def _style_pooling(self, d1, d2, eps=1e-5):
        N, C, _, _ = d1.size()
        N, C2, _, _ = d2.size()

        # branch one
        channel_mean = d1.view(N, C, -1).mean(dim=2, keepdim=True)
        channel_var = d1.view(N, C, -1).var(dim=2, keepdim=True) + eps
        channel_std = channel_var.sqrt()
        t1 = torch.cat((channel_mean, channel_std), dim=2)
        t1 = t1.view(N, C, 2, 1)

        # branch two
        channel_mean2 = d2.view(N, C2, -1).mean(dim=2, keepdim=True)
        channel_var2 = d2.view(N, C2, -1).var(dim=2, keepdim=True) + eps
        channel_std2 = channel_var2.sqrt()
        t2 = torch.cat((channel_mean2, channel_std2), dim=2)
        t2 = t2.view(N, C2, 2, 1)

        return t1, t2

    def forward(self, x):
        b, c, _, _ = x.size()

        x1 = x[:, :c // 2, :, :]
        x2 = x[:, c // 2:, :, :]

        d1 = self.conv1(x1)
        d1 = self.relu(self.bn1(d1))

        d2 = self.conv2(x2)
        d2 = self.relu(self.bn2(d2))

        t1, t2 = self._style_pooling(d1, d2)

        # B x C x 1 x 1
        g1 = self.sigmoid(self.fc1(t1))
        g2 = self.sigmoid(self.fc2(t2))
        g = torch.cat((g1, g2), dim=1)

        out = x * g
        return out


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1), torch.std(x, 1).unsqueeze(1)),
            dim=1
        )


class PSF(nn.Module):
    def __init__(self):
        super(PSF, self).__init__()
        self.compress = ChannelPool()

        self.sigmoid = nn.Sigmoid()

        self.spatial = BasicConv(3, 1, 1, stride=1, padding=0, relu=False)
        self.spatial2 = BasicConv(3, 1, 3, stride=1, padding=(3 - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        out = (self.spatial(x_compress) + self.spatial2(x_compress) + self.spatial2(x_compress)) / 3
        atten = self.sigmoid(out)
        # broadcasting
        return x * atten


class ADSR(nn.Module):
    def __init__(self, gate_channels, no_spatial=False):
        super(ADSR, self).__init__()
        self.ChannelGate = GSA(gate_channels)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = PSF()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, layer_idx, stride=1, downsample=None, use_cbam=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        if use_cbam:
            self.cbam = ADSR(planes)
        else:
            self.cbam = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if not self.cbam is None:
            out = self.cbam(out)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, layer_idx, stride=1, downsample=None, use_cbam=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        if use_cbam:
            self.cbam = ADSR(planes * 4)
        else:
            self.cbam = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if not self.cbam is None:
            out = self.cbam(out)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, network_type, num_classes, att_type=None):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.network_type = network_type
        # different model config between ImageNet and CIFAR
        if network_type == "ImageNet":
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.avgpool = nn.AdaptiveAvgPool2d(1)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)


        self.layer1 = self._make_layer(block, 64, 0, layers[0], att_type=att_type)
        self.layer2 = self._make_layer(block, 128, 1, layers[1], stride=2, att_type=att_type)
        self.layer3 = self._make_layer(block, 256, 2, layers[2], stride=2, att_type=att_type)
        self.layer4 = self._make_layer(block, 512, 3, layers[3], stride=2, att_type=att_type)

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        init.kaiming_normal_(self.fc.weight)
        for key in self.state_dict():
            if key.split('.')[-1] == "weight":
                if "conv" in key:
                    init.kaiming_normal_(self.state_dict()[key], mode='fan_out')
                if "bn" in key:
                    if "SpatialGate" in key:
                        self.state_dict()[key][...] = 0
                    else:
                        self.state_dict()[key][...] = 1
            elif key.split(".")[-1] == 'bias':
                self.state_dict()[key][...] = 0

    def _make_layer(self, block, planes, layer_index, blocks, stride=1, att_type=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, layer_index, stride, downsample, use_cbam=att_type == 'CBAM'))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, layer_index, use_cbam=att_type == 'CBAM'))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.network_type == "ImageNet":
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.network_type == "ImageNet":
            x = self.avgpool(x)
        else:
            x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def ResidualNet(network_type, depth, num_classes, att_type):
    assert network_type in ["ImageNet", "CIFAR10", "CIFAR100"], "network type should be ImageNet or CIFAR10 / CIFAR100"
    assert depth in [18, 34, 50, 101], 'network depth should be 18, 34, 50 or 101'

    if depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], network_type, num_classes, att_type)

    elif depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], network_type, num_classes, att_type)

    elif depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], network_type, num_classes, att_type)

    elif depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], network_type, num_classes, att_type)

    return model


def adsr_resnet18(num_classes=3):
    model = ResidualNet("ImageNet", 18, num_classes, 'CBAM')
    return model


def adsr_resnet34(num_classes=3):
    model = ResidualNet("ImageNet", 34, num_classes, 'CBAM')
    return model


def adsr_resnet50(num_classes=3):
    model = ResidualNet("ImageNet", 50, num_classes, 'CBAM')
    return model


def adsr_resnet101(num_classes=3):
    model = ResidualNet('ImageNet', 101, num_classes, 'CBAM')
    return model


if __name__ == '__main__':
    temp = torch.randn((2, 3, 224, 224))
    net = adsr_resnet50()
    out = net(temp)
    print(out)

