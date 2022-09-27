import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode
from spikingjelly.clock_driven import layer
from spikingjelly.clock_driven import surrogate  # surrogate function from 'Sigmoid' to 'ATan'
from q_module import *

def conv3x3(in_channels, out_channels, mode=None):
    if mode in ['bin', 'ter']:
        return nn.Sequential(
            layer.SeqToANNContainer(
                QConv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False, mode=mode),
                nn.BatchNorm2d(out_channels),
            ),
            MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan())
        )
    else:
        return nn.Sequential(
            layer.SeqToANNContainer(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels),
            ),
            MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan())
        )

def conv1x1(in_channels, out_channels, mode=None):
    # quantization mode is added
    if mode in ['bin', 'ter']:    
        return nn.Sequential(
            layer.SeqToANNContainer(
                QConv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False, mode=mode),
                nn.BatchNorm2d(out_channels),
            ),
            MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan())
        )
    else:
        return nn.Sequential(
            layer.SeqToANNContainer(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels),
            ),
            MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan())
        )

class SEWBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, connect_f=None, mode=None):
        super(SEWBlock, self).__init__()
        self.connect_f = connect_f
        self.conv = nn.Sequential(
            conv3x3(in_channels, mid_channels, mode=mode),
            conv3x3(mid_channels, in_channels, mode=mode),
        )

    def forward(self, x: torch.Tensor):
        out = self.conv(x)
        if self.connect_f == 'ADD':
            out += x
        elif self.connect_f == 'AND':
            out *= x
        elif self.connect_f == 'IAND':
            out = x * (1. - out)
        else:
            raise NotImplementedError(self.connect_f)

        return out

class PlainBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, mode=None):
        super(PlainBlock, self).__init__()
        self.conv = nn.Sequential(
            conv3x3(in_channels, mid_channels, mode=mode),
            conv3x3(mid_channels, in_channels, mode=mode),
        )

    def forward(self, x: torch.Tensor):
        return self.conv(x)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, mode=None):
        super(BasicBlock, self).__init__()
        self.conv = nn.Sequential(
            conv3x3(in_channels, mid_channels, mode=mode),

            layer.SeqToANNContainer(
                nn.Conv2d(mid_channels, in_channels, kernel_size=3, padding=1, stride=1, bias=False),
                nn.BatchNorm2d(in_channels),
            ),
        )
        self.sn = MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan())

    def forward(self, x: torch.Tensor):
        return self.sn(x + self.conv(x))


class ResNetN(nn.Module):
    def __init__(self, layer_list, num_classes, connect_f=None, mode=None):
        super(ResNetN, self).__init__()
        in_channels = 2
        conv = []
    
        for cfg_dict in layer_list:
            channels = cfg_dict['channels']

            if 'mid_channels' in cfg_dict:
                mid_channels = cfg_dict['mid_channels']
            else:
                mid_channels = channels
            
            ### First layer is not quantized. ###
            if in_channels != channels and in_channels == 2:
                if cfg_dict['up_kernel_size'] == 3:
                    conv.append(conv3x3(in_channels, channels))
                elif cfg_dict['up_kernel_size'] == 1:
                    conv.append(conv1x1(in_channels, channels))
                else:
                    raise NotImplementedError
                    
            elif in_channels != channels:
                if cfg_dict['up_kernel_size'] == 3:
                    conv.append(conv3x3(in_channels, channels, mode=mode))
                elif cfg_dict['up_kernel_size'] == 1:
                    conv.append(conv1x1(in_channels, channels, mode=mode))
                else:
                    raise NotImplementedError

            in_channels = channels

            if 'num_blocks' in cfg_dict:
                num_blocks = cfg_dict['num_blocks']
                if cfg_dict['block_type'] == 'sew':
                    for _ in range(num_blocks):
                        conv.append(SEWBlock(in_channels, mid_channels, connect_f, mode))
                elif cfg_dict['block_type'] == 'plain':
                    for _ in range(num_blocks):
                        conv.append(PlainBlock(in_channels, mid_channels, mode))
                elif cfg_dict['block_type'] == 'basic':
                    for _ in range(num_blocks):
                        conv.append(BasicBlock(in_channels, mid_channels, mode))
                else:
                    raise NotImplementedError

            if 'k_pool' in cfg_dict:
                k_pool = cfg_dict['k_pool']
                conv.append(layer.SeqToANNContainer(nn.MaxPool2d(k_pool, k_pool)))

        conv.append(nn.Flatten(2))

        self.conv = nn.Sequential(*conv)

        with torch.no_grad():
            x = torch.zeros([1, 1, 128, 128])
            for m in self.conv.modules():
                if isinstance(m, nn.MaxPool2d):
                    x = m(x)
            out_features = x.numel() * in_channels
        
        ### Last layer is not quantized. ###
        self.out = nn.Linear(out_features, num_classes, bias=True)

    def forward(self, x: torch.Tensor):
        x = x.permute(1, 0, 2, 3, 4)  # [T, N, 2, *, *]
        x = self.conv(x)
        return self.out(x.mean(0))

def SEWResNet(connect_f, mode):
    layer_list = [
        {'channels': 64, 'up_kernel_size': 3, 'mid_channels': 64, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},
        {'channels': 64, 'up_kernel_size': 1, 'mid_channels': 64, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},
        {'channels': 64, 'up_kernel_size': 1, 'mid_channels': 64, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},
        {'channels': 64, 'up_kernel_size': 1, 'mid_channels': 64, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},
        {'channels': 128, 'up_kernel_size': 3, 'mid_channels': 128, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},
        {'channels': 128, 'up_kernel_size': 1, 'mid_channels': 128, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},
        {'channels': 128, 'up_kernel_size': 1, 'mid_channels': 128, 'num_blocks': 1, 'block_type': 'sew', 'k_pool': 2},
    ]
    num_classes = 10
    return ResNetN(layer_list, num_classes, connect_f, mode)