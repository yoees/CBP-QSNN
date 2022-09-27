import numpy as np
import torch.nn as nn


class StraightThrough(nn.Module):
    def __init__(self, channel_num: int = 1):
        super().__init__()

    def forward(self, input):
        return input


class AvgPoolConv_q(nn.Conv2d):
    def __init__(self, kernel_size=2, stride=2, input_channel=64, padding=0, freeze_avg=True):
        super().__init__(input_channel, input_channel, kernel_size, padding=padding, stride=stride,
                         groups=input_channel, bias=False)
        # init the weight to make them equal to 1/k/k
        #self.set_weight_to_avg()
        self.freeze = freeze_avg
        self.relu = nn.ReLU(inplace=True)

    def forward(self, *inputs):
        #self.set_weight_to_avg()
        x = super().forward(*inputs)
        return self.relu(x)

    #def set_weight_to_avg(self):
    #    self.weight.data.fill_(1).div_(self.kernel_size[0] * self.kernel_size[1])

    