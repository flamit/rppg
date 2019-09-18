## Created by: Zilin Gao
## Implementation of "Global Second-order Pooling Convolutional Networks", Zilin Gao, Jiangtao Xie, Qilong Wang, Peihua Li. IN CVPR2019.
## Copyright (c) 2019 Zilin Gao, Jiangtao Xie, Qilong Wang and Peihua Li.
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import models.gsop.mpnconv as MPNCOV

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


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
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, attention='0', att_dim=128):
        super(Bottleneck, self).__init__()
        self.dimDR = att_dim
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.relu_normal = nn.ReLU(inplace=False)
        if attention is '1':
            if planes > 64:
                DR_stride = 1
            else:
                DR_stride = 2

            self.ch_dim = att_dim
            self.conv_for_DR = nn.Conv2d(planes * self.expansion, self.ch_dim, kernel_size=1, stride=DR_stride,
                                         bias=True)
            self.bn_for_DR = nn.BatchNorm2d(self.ch_dim)
            self.row_bn = nn.BatchNorm2d(self.ch_dim)
            self.row_conv_group = nn.Conv2d(self.ch_dim, 4 * self.ch_dim, kernel_size=(self.ch_dim, 1),
                                            groups=self.ch_dim, bias=True)
            self.fc_adapt_channels = nn.Conv2d(4 * self.ch_dim, planes * self.expansion, kernel_size=1, groups=1,
                                               bias=True)
            self.sigmoid = nn.Sigmoid()

        self.downsample = downsample
        self.stride = stride
        self.attention = attention

    def chan_att(self, out):
        # NxCxHxW
        out = self.relu_normal(out)
        out = self.conv_for_DR(out)
        out = self.bn_for_DR(out)
        out = self.relu(out)

        out = MPNCOV.CovpoolLayer(out)  # Nxdxd
        out = out.view(out.size(0), out.size(1), out.size(2), 1).contiguous()  # Nxdxdx1

        out = self.row_bn(out)
        out = self.row_conv_group(out)  # Nx512x1x1

        out = self.fc_adapt_channels(out)  # NxCx1x1
        out = self.sigmoid(out)  # NxCx1x1

        return out

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
        if self.attention is '1':
            pre_att = out
            att = self.chan_att(out)
            out = pre_att * att

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, att_position, att_dim, GSoP_mode, num_classes=1000):
        self.inplanes = 64
        self.GSoP_mode = GSoP_mode
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], att_position=att_position[0], att_dim=att_dim)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, att_position=att_position[1], att_dim=att_dim)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, att_position=att_position[2], att_dim=att_dim)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, att_position=att_position[3], att_dim=att_dim)
        if GSoP_mode == 1:
            self.avgpool = nn.AvgPool2d(14, stride=1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)
            print("GSoP-Net1 generating...")
        else:
            self.isqrt_dim = 256
            self.layer_reduce = nn.Conv2d(512 * block.expansion, self.isqrt_dim, kernel_size=1, stride=1, padding=0,
                                          bias=False)
            self.layer_reduce_bn = nn.BatchNorm2d(self.isqrt_dim)
            self.layer_reduce_relu = nn.ReLU(inplace=True)
            self.fc = nn.Linear(int(self.isqrt_dim * (self.isqrt_dim + 1) / 2), num_classes)
            print("GSoP-Net2 generating...")

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, att_position=[1], att_dim=128):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, att_position[0], att_dim=att_dim))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, attention=att_position[i], att_dim=att_dim))
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

        if self.GSoP_mode == 1:
            x = self.avgpool(x)
        else:
            x = self.layer_reduce(x)
            x = self.layer_reduce_bn(x)
            x = self.layer_reduce_relu(x)

            x = MPNCOV.CovpoolLayer(x)
            x = MPNCOV.SqrtmLayer(x, 3)
            x = MPNCOV.TriuvecLayer(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet50gsop(pretrained=False, att_position=[[], [], [], []], att_dim=128, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], att_position, att_dim, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model
