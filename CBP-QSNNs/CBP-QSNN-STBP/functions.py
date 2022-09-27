import torch
import torch.nn.functional as F

### Surrogate gradient function (box car function) ###
class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, thresh, lens):
        ctx.thresh = thresh
        ctx.lens = lens
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        thresh = ctx.thresh
        lens = ctx.lens
        grad_input = grad_output.clone()   
        temp = abs(input - thresh) < lens
        return grad_input * temp.float() / (2 * lens), None, None

act_fun = ActFun.apply


### LIF neuron (hard reset) ###
def mem_update(decay, thresh, lens, ops, x, mem, spike):
    mem = mem * decay * (1. - spike) + ops(x)
    spike = act_fun(mem, thresh, lens) 
    return mem, spike