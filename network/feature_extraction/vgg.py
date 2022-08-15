import torch.nn as nn
import torchvision.models as models

__all__ = ['vgg11', 'vgg16', 'vgg19']


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.net = None

    def forward(self, x):
        return self.avg_pool(self.net(x).permute(0, 3, 1, 2)).squeeze(3)


class VGG19(VGG):
    def __init__(self, input_channel):
        super(VGG19, self).__init__()
        layers = list(models.vgg19().features.children())[:-3]
        layers[0] = nn.Conv2d(input_channel, 64, kernel_size=(3, 3), padding=1)
        layers[32] = nn.Conv2d(512, 512, kernel_size=(2, 2), padding=0)
        for i in [18, 27]:
            layers[i] = nn.MaxPool2d((2, 1), (2, 1))
        for i in [30, 28, 25, 23, 21, 19]:
            layers[i] = nn.Conv2d(layers[i].in_channels, 512, kernel_size=(3, 3), padding=1, bias=False)
            layers.insert(i + 1, nn.BatchNorm2d(512))
        self.net = nn.Sequential(*layers)


class VGG16(VGG):
    def __init__(self, input_channel):
        super(VGG16, self).__init__()
        layers = list(models.vgg16().features.children())[:-3]
        layers[0] = nn.Conv2d(input_channel, 64, kernel_size=(3, 3), padding=1)
        layers[26] = nn.Conv2d(512, 512, kernel_size=(2, 2), padding=0)
        for i in [16, 23]:
            layers[i] = nn.MaxPool2d((2, 1), (2, 1))
        for i in [24, 21, 19, 17]:
            layers[i] = nn.Conv2d(layers[i].in_channels, 512, kernel_size=(3, 3), padding=1, bias=False)
            layers.insert(i + 1, nn.BatchNorm2d(512))
        self.net = nn.Sequential(*layers)


class VGG11(VGG):
    def __init__(self, input_channel):
        super(VGG11, self).__init__()
        layers = list(models.vgg11().features.children())[:-3]
        layers[0] = nn.Conv2d(input_channel, 64, kernel_size=(3, 3), padding=1)
        layers[16] = nn.Conv2d(512, 512, kernel_size=(2, 2), padding=0)
        for i in [10, 15]:
            layers[i] = nn.MaxPool2d((2, 1), (2, 1))
        for i in [13, 11]:
            layers[i] = nn.Conv2d(layers[i].in_channels, 512, kernel_size=(3, 3), padding=1, bias=False)
            layers.insert(i + 1, nn.BatchNorm2d(512))
        self.net = nn.Sequential(*layers)


def vgg11(opt):
    return VGG11(opt.nc)


def vgg16(opt):
    return VGG16(opt.nc)


def vgg19(opt):
    return VGG19(opt.nc)
