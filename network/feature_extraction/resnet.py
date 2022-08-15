import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

__all__ = ['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=(1, 1), pooling_stride=None):
        super(BasicBlock, self).__init__()
        self.pooling = False
        if pooling_stride is not None:
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=pooling_stride, padding=0)
            self.pooling = True
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=(3, 3), stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        if self.pooling:
            x = self.maxpool(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, input_channel, output_channel, block, num_blocks):
        super(ResNet, self).__init__()
        out_channels = [output_channel // 16, output_channel // 8, output_channel // 4, output_channel // 2]
        self.in_planes = output_channel // 8
        self.conv1 = nn.Conv2d(input_channel, out_channels[0], kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels[0])
        self.conv2 = nn.Conv2d(out_channels[0], out_channels[1], kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels[1])
        self.layer1 = self._make_layer(block, out_channels[1], num_blocks[0], stride=1, pooling_stride=2)
        self.layer2 = self._make_layer(block, out_channels[2], num_blocks[1], stride=1, pooling_stride=2)
        self.layer3 = self._make_layer(block, out_channels[3], num_blocks[2], stride=1, pooling_stride=(2, 1))
        self.layer4 = self._make_layer(block, output_channel, num_blocks[2], stride=(2, 1))
        self.conv5 = nn.Conv2d(output_channel, output_channel, kernel_size=(1, 1), stride=(2, 1), padding=0, bias=False)
        self.bn5 = nn.BatchNorm2d(output_channel)
        self.avg_pool = nn.AdaptiveAvgPool2d((None, 1))

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, pooling_stride=None):
        strides = [stride] + [1] * (num_blocks - 1)
        pooling_stride = [pooling_stride] + [None] * (num_blocks - 1)
        layers = []
        for i, stride in enumerate(strides):
            layers.append(block(self.in_planes, planes, stride, pooling_stride[i]))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.bn5(self.conv5(out)))
        out = self.avg_pool(out.permute(0, 3, 1, 2)).squeeze(3)
        return out


def resnet20(opt):
    return ResNet(opt.nc, opt.n_features, BasicBlock, [3, 3, 3])


def resnet32(opt):
    return ResNet(opt.nc, opt.n_features, BasicBlock, [5, 5, 5])


def resnet44(opt):
    return ResNet(opt.nc, opt.n_features, BasicBlock, [7, 7, 7])


def resnet56(opt):
    return ResNet(opt.nc, opt.n_features, BasicBlock, [9, 9, 9])


def resnet110(opt):
    return ResNet(opt.nc, opt.n_features, BasicBlock, [18, 18, 18])


def resnet1202(opt):
    return ResNet(opt.nc, opt.n_features, BasicBlock, [200, 200, 200])

