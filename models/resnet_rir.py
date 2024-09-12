import torch.nn as nn
import torch
import torch.nn.functional as F


class RIR(nn.Module):

    def __init__(self, in_channels, out_channels, r=16, L=32, M=3):
        super(RIR, self).__init__()

        d = max(in_channels // r, L)
        self.M = M

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.se = nn.Sequential(nn.Conv2d(in_channels, d, 1, bias=False),
                                nn.ReLU(inplace=True))

        self.out_channels = out_channels

        # self
        self.up_conv = nn.Conv2d(in_channels, in_channels, 1, 1, bias=False)
        self.center_conv = nn.Conv2d(in_channels, in_channels, 1, 1, bias=False)
        self.down_conv = nn.Conv2d(in_channels, in_channels, 1, 1, bias=False)

        self.fc2 = nn.Conv2d(d, out_channels, 1, 1, bias=False)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size, c, height, width = x.size()

        x1 = x[:, :, :height // 3, :]  # up nuclear region
        x2 = x[:, :, height // 3: height * 2 // 3, :]
        x3 = x[:, :, height * 2 // 3:, :]

        up_y = self.avg_pool(x1)
        center_y = self.avg_pool(x2)
        down_y = self.avg_pool(x3)

        y = up_y + down_y + center_y

        # gap = self.global_pool(input)
        z = self.se(y)

        a_b_c = self.fc2(z)

        a_b_c = a_b_c.reshape(batch_size, self.M, self.out_channels // self.M, -1)
        a_b_c = self.softmax(a_b_c)

        # the part of selection
        a_b_c = list(a_b_c.chunk(self.M, dim=1))  # split to a b, and c
        a_b_c = list(map(lambda x: x.reshape(batch_size, self.out_channels // self.M, 1, 1), a_b_c))

        a = a_b_c[0]
        b = a_b_c[1]
        c = a_b_c[2]

        up_region = a * x1
        center_region = b * x2
        down_region = c * x3

        y = torch.cat([up_region, center_region, down_region], dim=2)

        return y


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


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
        self.rir = RIR(planes, planes * 3)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.rir(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.rir = RIR(planes * self.expansion, planes * self.expansion * 3)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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

        out = self.rir(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=5, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=1,
        #                        bias=False)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
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

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
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
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def rir_resnet18(num_classes=4):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes)
    return model


def rir_resnet34(num_classes=3):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes)
    return model


def rir_resnet50(num_classes=5):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes)
    return model


def rir_resnet101(num_classes=3):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes)
    return model


def rir_resnet152(num_classes=3):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], num_classes)
    return model


if __name__ == '__main__':
    temp = torch.randn((1, 3, 224, 224))

    net = rir_resnet50()
    out = net(temp)
    print(out)
