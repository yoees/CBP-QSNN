import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

### This code is from https://github.com/dooseokjeong/CBP/blob/main/module.py ###

### STE (Straight Through Estimator) ###
class Quantizetotal(Function):
    @staticmethod
    def forward(ctx, weight, scale, b, df):
        ctx.save_for_backward(weight, scale)
        out = df[0] * (torch.sign(weight.detach() - b[0] * scale.detach()) + 1) / 2 
        for i in range(1, len(b)):
            out.data += df[i] * (torch.sign(weight.detach() - b[i] * scale.detach()) + 1) / 2 
        out.data = (out.data - 1) * scale.detach()
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        # backward pass uses real-valued weight w under quantization
        weight, scale = ctx.saved_tensors
        return grad_output, None, None, None

### Linear layer (quantized weight) ###
class QLinear(nn.Linear):

    def __init__(self, in_feature, out_feature, bias=False, mode = 'bin'):
        super(QLinear, self).__init__(in_feature,out_feature, bias)
        self.scale = torch.nn.Parameter(torch.FloatTensor([self.weight.abs().mean()]).cuda())
        self.qweight = 0

        if mode == 'bin':  # binary
            self.factor = [-1, 1]
        elif mode =='ter':  # ternary
            self.factor = [-1, 0, 1]

        self.df = []   # difference
        self.b = []    # M set (median)
        for p in range(len(self.factor)-1):
            self.df+=[self.factor[p+1]-self.factor[p]]
            self.b += [(self.factor[p+1]+self.factor[p])/2]

    def forward(self, input):
        self.qweight = Quantizetotal().apply(self.weight,self.scale,self.b,self.df)
        output = F.linear(input, self.qweight, self.bias)

        return output

### Conv2d layer (quantized weight) ###    
class QConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, mode = 'bin'):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.scale = torch.nn.Parameter(torch.FloatTensor([self.weight.abs().mean()]).cuda())
        self.qweight = 0

        if mode == 'bin':  # binary
            self.factor = [-1, 1]
        elif mode =='ter':  # ternary
            self.factor = [-1, 0, 1]

        self.df = []  # difference
        self.b = []   # M set (median)
        for p in range(len(self.factor)-1):
            self.df+=[self.factor[p+1]-self.factor[p]]
            self.b += [(self.factor[p+1]+self.factor[p])/2]

    def forward(self, input):
        self.qweight = Quantizetotal().apply(self.weight,self.scale,self.b,self.df)
        output = F.conv2d(input, self.qweight, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)
        return output
    

### Constraint function ###
# %% maximum to max / minimum to min
class constraint(Function):
    @staticmethod
    def forward(ctx, weight, scale, factor, b, ucs):
        ctx.save_for_backward(weight)
        ctx.scale = scale   # scale factor
        ctx.factor = factor # Q set
        ctx.b = b           # median
        ctx.ucs = ucs       # unconstrained window
        
        out = torch.max((weight - scale.detach() * factor[0]) * -2, torch.zeros(1).cuda())        
        for p in range(len(b)):
            qlow = factor[p] * scale
            qhigh = factor[p+1] * scale
            mean = b[p] * scale
            line = torch.max(torch.min((weight-qlow)/(mean-qlow), (weight-qhigh)/(mean-qhigh))*(qhigh-qlow),
                                torch.zeros(1).cuda())
            line.data[line.detach() > (qhigh-qlow)*ucs] = 0 # apply unconstrained window
            out.data += line
        out.data += torch.max((weight - scale.detach() * factor[-1]) * 2, torch.zeros(1).cuda())
        return out
    
    @staticmethod
    def backward(ctx, grad_input):
        weight = ctx.saved_tensors[0]
        scale = ctx.scale
        factor = ctx.factor
        b = ctx.b
        ucs = ctx.ucs
        
        grad_input.data[weight < factor[0] * scale] *= -2
        grad_input.data[weight >= factor[-1] * scale] *= 2
        for p in range(len(b)):
            qlow = factor[p] * scale
            qhigh = factor[p+1] * scale
            mean = b[p] * scale
            rlow = qlow + (mean - qlow) * ucs
            rhigh = qhigh + (mean - qhigh) * ucs

            grad_input.data[torch.logical_and(weight >= qlow, weight < rlow)] *= (qhigh-qlow)/(mean-qlow)
            grad_input.data[torch.logical_and(weight < qhigh, weight >= rhigh)] *= (qhigh-qlow)/(mean-qhigh)
            grad_input.data[torch.logical_and(weight >= rlow, weight < rhigh)] = 0 # apply unconstrained window
            
        return grad_input, None, None, None, None