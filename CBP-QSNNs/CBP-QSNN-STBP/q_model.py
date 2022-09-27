import torch 
import torch.nn as nn 
from q_module import *
from functions import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def conv(in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, mode=None):
    if mode in ['bin', 'ter']:
        return QConv2d(in_planes, out_planes, kernel_size, stride=stride,
                     padding=padding, groups=groups, bias=bias, dilation=dilation, mode = mode)
    else:
        print('entered mode not exist. conventional convolution will be set')
        return nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride,
                     padding=padding, groups=groups, bias=bias, dilation=dilation)

def linear(in_features, out_features, bias=True, mode=None):
    if mode in ['bin', 'ter']:
        return QLinear(in_features, out_features, bias, mode = mode) 

    else:
        print('entered mode not exist, conventional linear function will be set')
        return nn.Linear(in_features, out_features, bias)

    
class Q_Conv_SNN(nn.Module):
    def __init__(self, outs, decay, thresh, lens, batch_size, timesteps, mode):
        super(Q_Conv_SNN, self).__init__()
        
        self.outs = outs # the number of output neurons
        self.decay = decay
        self.thresh = thresh
        self.lens = lens
        self.batch_size = batch_size
        self.T = timesteps
        self.mode = mode # binary, ternary
        
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1)  # do not quantize 1st layer
        self.pool1 = conv(128, 256, kernel_size=3, stride=2, padding=1, mode=mode)
        
        self.conv2 = conv(256, 256, kernel_size=3, stride=1, padding=1, mode=mode)
        self.pool2 = conv(256, 512, kernel_size=3, stride=2, padding=1, mode=mode)
        
        self.conv3 = conv(512, 512, kernel_size=3, stride=1, padding=1, mode=mode)
        self.conv4 = conv(512, 1024, kernel_size=3, stride=1, padding=1, mode=mode)
        self.pool3 = conv(1024, 2048, kernel_size=3, stride=2, padding=1, mode=mode)       
        
        self.fc1 = linear(2048*4*4, 1024, mode=mode)
        self.fc2 = linear(1024, 512, mode=mode)
        self.fc3 = nn.Linear(512, outs)  # do not quantize last layer

    def forward(self, input):
        
        c1_mem = c1_spike = torch.zeros(self.batch_size, 128, 32, 32, device=device)
        p1_mem = p1_spike = torch.zeros(self.batch_size, 256, 16, 16, device=device)
        
        c2_mem = c2_spike = torch.zeros(self.batch_size, 256, 16, 16, device=device)
        p2_mem = p2_spike = torch.zeros(self.batch_size, 512, 8, 8, device=device)
        
        c3_mem = c3_spike = torch.zeros(self.batch_size, 512, 8, 8, device=device)
        c4_mem = c4_spike = torch.zeros(self.batch_size, 1024, 8, 8, device=device)
        p3_mem = p3_spike = torch.zeros(self.batch_size, 2048, 4, 4, device=device)
        
        h1_mem = h1_spike = torch.zeros(self.batch_size, 1024, device=device)
        h2_mem = h2_spike = torch.zeros(self.batch_size, 512, device=device)
        h3_mem = h3_spike = h3_sumspike = torch.zeros(self.batch_size, self.outs, device=device)

        for step in range(self.T): # simulation time steps

            c1_mem, c1_spike = mem_update(self.decay, self.thresh, self.lens, self.conv1, input, c1_mem, c1_spike)
            p1_mem, p1_spike = mem_update(self.decay, self.thresh, self.lens, self.pool1, c1_spike, p1_mem, p1_spike)
            
            c2_mem, c2_spike = mem_update(self.decay, self.thresh, self.lens, self.conv2, p1_spike, c2_mem, c2_spike)
            p2_mem, p2_spike = mem_update(self.decay, self.thresh, self.lens, self.pool2, c2_spike, p2_mem, p2_spike)
            
            c3_mem, c3_spike = mem_update(self.decay, self.thresh, self.lens, self.conv3, p2_spike, c3_mem, c3_spike)
            c4_mem, c4_spike = mem_update(self.decay, self.thresh, self.lens, self.conv4, c3_spike, c4_mem, c4_spike)
            p3_mem, p3_spike = mem_update(self.decay, self.thresh, self.lens, self.pool3, c4_spike, p3_mem, p3_spike)

            h1_mem, h1_spike = mem_update(self.decay, self.thresh, self.lens, self.fc1, p3_spike.view(self.batch_size, -1) , h1_mem, h1_spike)
            h2_mem, h2_spike = mem_update(self.decay, self.thresh, self.lens, self.fc2, h1_spike, h2_mem, h2_spike)
            h3_mem, h3_spike = mem_update(self.decay, self.thresh, self.lens, self.fc3, h2_spike, h3_mem, h3_spike)
            
            h3_sumspike += h3_spike

        outputs = h3_sumspike / self.T
        
        return outputs