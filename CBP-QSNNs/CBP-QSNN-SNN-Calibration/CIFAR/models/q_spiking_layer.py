import copy
import math
import torch
from torch.autograd import Function
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import Callable, Tuple, List, Union, Dict, cast
from torch.utils.data import DataLoader
from CIFAR.models.q_utils import StraightThrough, AvgPoolConv_q
from distributed_utils.dist_helper import allaverage
from CIFAR.models.q_module import * 


### Surrogate gradient function ###
class SpikeAct(Function):
    @staticmethod
    def forward(ctx, input, threshold, alpha):
        x = input - threshold 
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        output = (x >= 0.).float()
        return output
        
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_x = ctx.alpha / 2 / (1 + (math.pi / 2 * ctx.alpha * x).pow_(2)) * grad_input
        return grad_x, None, None
    
spikeAct = SpikeAct.apply

# ----------- SpikeModule (nq)-----------

class SpikeModule_nq(nn.Module):
    def __init__(self, sim_length: int, conv: Union[nn.Conv2d, nn.Linear], enable_shift: bool = True):
        super(SpikeModule_nq, self).__init__()
        
        self.conv = conv
        self.threshold = None
        self.mem_pot = 0
        self.mem_pot_init = 0
            
        self.use_spike = False
        self.enable_shift = enable_shift
        self.sim_length = sim_length
        self.cur_t = 0
        self.relu = StraightThrough()
        
    def forward(self, input: torch.Tensor):      
        if self.use_spike:
            self.cur_t += 1
            x = self.conv(input)
            if self.enable_shift is True and self.threshold is not None:
                x = x + self.threshold * 0.5 / self.sim_length
            self.mem_pot = self.mem_pot + x
            spike = spikeAct(self.mem_pot, self.threshold, 2)
            self.mem_pot -= spike
            return spike

    def init_membrane_potential(self):
        self.mem_pot = self.mem_pot_init if isinstance(self.mem_pot_init, int) else self.mem_pot_init.clone()
        self.cur_t = 0


class SpikeModule_q(nn.Module):
    def __init__(self, sim_length: int, conv: Union[QConv2d, QLinear], enable_shift: bool = True):
        super(SpikeModule_q, self).__init__()
        
        self.conv = conv 
        self.threshold = None
        self.mem_pot = 0
        self.mem_pot_init = 0
        
        self.use_spike = False
        self.enable_shift = enable_shift
        self.sim_length = sim_length
        self.cur_t = 0
        self.relu = StraightThrough()
        
    def forward(self, input: torch.Tensor):      
        if self.use_spike:
            self.cur_t += 1
            x = self.conv(input)
            if self.enable_shift is True and self.threshold is not None:
                x = x + self.threshold * 0.5 / self.sim_length
            self.mem_pot = self.mem_pot + x
            spike = spikeAct(self.mem_pot, self.threshold, 2)
            self.mem_pot -= spike
            return spike

    def init_membrane_potential(self):
        self.mem_pot = self.mem_pot_init if isinstance(self.mem_pot_init, int) else self.mem_pot_init.clone()
        self.cur_t = 0


class SpikeModel_q(nn.Module):

    def __init__(self, model: nn.Module, sim_length: int, specials: dict = {}):
        super().__init__()
        self.model = model
        self.specials = specials
        self.spike_module_refactor(self.model, sim_length)
        self.use_spike = False

        assert sim_length > 0, "SNN does not accept negative simulation length"
        self.T = sim_length

    def spike_module_refactor(self, module: nn.Module, sim_length: int, prev_module=None):
        """
        Recursively replace the normal conv2d to SpikeConv2d
        :param module: nn.Module with nn.Conv2d or nn.Linear in its children
        :param sim_length: simulation length, aka total time steps
        :param prev_module: use this to add relu to prev_spikemodule
        """
        prev_module = prev_module
        for name, immediate_child_module in module.named_children():
            if type(immediate_child_module) in self.specials:
                setattr(module, name, self.specials[type(immediate_child_module)]
                                                        (immediate_child_module, sim_length=sim_length))
            elif isinstance(immediate_child_module, nn.Conv2d) and not isinstance(immediate_child_module, AvgPoolConv_q):
                if 'factor' in dir(immediate_child_module):
                    setattr(module, name, SpikeModule_q(sim_length=sim_length, conv=immediate_child_module))
                    prev_module = getattr(module, name)
                else:
                    setattr(module, name, SpikeModule_nq(sim_length=sim_length, conv=immediate_child_module))
                    prev_module = getattr(module, name)
            elif isinstance(immediate_child_module, (nn.ReLU, nn.ReLU6)):
                if prev_module is not None:
                    prev_module.add_module('relu', immediate_child_module)
                    setattr(module, name, StraightThrough())
                else:
                    continue
            elif isinstance(immediate_child_module, AvgPoolConv_q):
                relu = immediate_child_module.relu
                setattr(module, name, SpikeModule_nq(sim_length=sim_length, conv=immediate_child_module))
                getattr(module, name).add_module('relu', relu)
            else:
                prev_module = self.spike_module_refactor(immediate_child_module, sim_length=sim_length, prev_module=prev_module)

        return prev_module

    def set_spike_state(self, use_spike: bool = True):
        self.use_spike = use_spike
        for m in self.model.modules():
            if isinstance(m, (SpikeModule_nq, SpikeModule_q)):
                m.use_spike = use_spike

    def init_membrane_potential(self):
        for m in self.model.modules():
            if isinstance(m, (SpikeModule_nq, SpikeModule_q)):
                m.init_membrane_potential()

    def forward(self, input):
        if self.use_spike:
            self.init_membrane_potential()
            out = 0
            for sim in range(self.T):
                out += self.model(input)
                
        return out


    

    
                           
            
                            
    
                
            
        
                
        
        
            
        
    


