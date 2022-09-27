import torch 
import torch.nn as nn 
from functions import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Conv_SNN(nn.Module):
    def __init__(self, outs, decay, thresh, lens, batch_size, timesteps):
        super(Conv_SNN, self).__init__()
        
        self.outs = outs # the number of output neurons
        self.decay = decay
        self.thresh = thresh
        self.lens = lens
        self.batch_size = batch_size
        self.T = timesteps
        
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1) # stride 2 conv.
        
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1) # stride 2 conv.
        
        self.conv3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.Conv2d(1024, 2048, kernel_size=3, stride=2, padding=1) # stride 2 conv.        
        
        self.fc1 = nn.Linear(2048*4*4, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, outs)

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