import torch
import torch.nn as nn
import math
from CIFAR.models.q_utils import AvgPoolConv_q
from CIFAR.models.q_module import * 


cfg = {
    'VGG11': [16, 'M', 32, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def conv(in_planes, out_planes, kernel_size, stride=1, padding = 1,  groups=1, dilation=1, mode=None, bias = False):
    if mode in ['bin', 'ter']:
        return QConv2d(in_planes, out_planes, kernel_size, stride=stride,
                     padding=padding, groups=groups, bias=bias, dilation=dilation, mode = mode)
    else:
        print('entered mode not exist. conventional convolution will be set')
        return nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride,
                     padding=padding, groups=groups, bias=bias, dilation=dilation)

def linear(in_features, out_features, bias = False, mode = None):
    if mode in ['bin', 'ter']:
        return QLinear(in_features, out_features, bias, mode = mode) 
    else:
        print('entered mode not exist, conventional linear function will be set')
        return nn.Linear(in_features, out_features, bias)

class VGG_q(nn.Module):
    def __init__(self, vgg_name, num_class=100, use_bn=True, mode=None):
        super(VGG_q, self).__init__()
        self.mode = mode
        self.features = self._make_layers(cfg[vgg_name], use_bn)
        self.classifier = nn.Linear(512, num_class)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out) # Last layer is not quantized
        return out

    def _make_layers(self, cfg, use_bn=True):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [AvgPoolConv_q(kernel_size=2, stride=2, input_channel=in_channels)]
            else:
                if in_channels == 3: # First layer is not quantized
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                               nn.BatchNorm2d(x) if use_bn else nn.Dropout(0.25),
                               nn.ReLU(inplace=True)]
                    in_channels = x
                else:
                    layers += [conv(in_channels, x, kernel_size=3, padding=1, mode=self.mode, bias=True),
                               nn.BatchNorm2d(x) if use_bn else nn.Dropout(0.25),
                               nn.ReLU(inplace=True)]
                    in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)