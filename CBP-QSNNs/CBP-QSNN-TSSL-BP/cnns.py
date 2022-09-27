import torch
import torch.nn as nn
import layers.q_module as module
import layers.conv as conv
import layers.pooling as pooling
import layers.dropout as dropout
import layers.linear as linear
import functions.loss_f as f
import global_v as glv


class Network(nn.Module):
    def __init__(self, network_config, layers_config, input_shape):
        super(Network, self).__init__()
        self.layers = []
        self.network_config = network_config
        self.layers_config = layers_config
        parameters = []  # overall weights
        
        nq_layers = []   # weight to be quantized layer num list
        q_layers = []    # weight not to be quantized layer num list
        
        self.factor_ = []   # factors of each quantized layers
        self.b_ = []        # b of each quantized layers (median)
        self.scale_ = []    # scale factor of each quantized layers
        
        param_size = 0
        layer_num = 0   
        
        print("Network Structure:")
        for key in layers_config:
            c = layers_config[key]
            if c['type'] == 'conv':
                if (network_config['mode'] is not None) and (c['qtype'] == 'q'):  # quantize
                    self.layers.append(conv.QConvLayer(network_config, c, key, input_shape))
                    input_shape = self.layers[-1].out_shape
                    parameters.append(self.layers[-1].get_parameters())
                    
                    param_size += self.layers[-1].get_parameters().numel()
                    self.factor_ += [self.layers[-1].factor]
                    self.b_ += [self.layers[-1].b]
                    self.scale_ += [self.layers[-1].scale]
                    q_layers.append(layer_num)
                    
                else:  # not quantize
                    self.layers.append(conv.ConvLayer(network_config, c, key, input_shape))
                    input_shape = self.layers[-1].out_shape
                    parameters.append(self.layers[-1].get_parameters())
                    nq_layers.append(layer_num)
                layer_num += 1
                
            elif c['type'] == 'linear':
                if (network_config['mode'] is not None) and (c['qtype'] == 'q'):  # quantize
                    self.layers.append(linear.QLinearLayer(network_config, c, key, input_shape))
                    input_shape = self.layers[-1].out_shape
                    parameters.append(self.layers[-1].get_parameters())
                    
                    param_size += self.layers[-1].get_parameters().numel()
                    self.factor_ += [self.layers[-1].factor]
                    self.b_ += [self.layers[-1].b]
                    self.scale_ += [self.layers[-1].scale]
                    q_layers.append(layer_num)
                                   
                else:  # not quantize
                    self.layers.append(linear.LinearLayer(network_config, c, key, input_shape))
                    input_shape = self.layers[-1].out_shape
                    parameters.append(self.layers[-1].get_parameters())
                    nq_layers.append(layer_num)
                layer_num += 1
                
            elif c['type'] == 'pooling':
                self.layers.append(pooling.PoolLayer(network_config, c, key, input_shape))
                input_shape = self.layers[-1].out_shape
                
            elif c['type'] == 'dropout':
                self.layers.append(dropout.DropoutLayer(c, key))
                
            else:
                raise Exception('Undefined layer type. It is: {}'.format(c['type']))
                
        self.my_parameters = nn.ParameterList(parameters)
        self.nq_layers_list = nq_layers
        self.q_layers_list = q_layers
        self.param_size_total = param_size
        print("-----------------------------------------")

    def forward(self, spike_input, epoch, is_train):
        spikes = f.psp(spike_input, self.network_config)
        skip_spikes = {}
        assert self.network_config['model'] == "LIF"
        
        for l in self.layers:
            if l.type == "dropout":
                if is_train:
                    spikes = l(spikes)
            elif self.network_config["rule"] == "TSSLBP":
                spikes = l.forward_pass(spikes, epoch)
            else:
                raise Exception('Unrecognized rule type. It is: {}'.format(self.network_config['rule']))
        return spikes

    def get_parameters(self):
        if self.network_config['mode'] is not None:
            nq_parameters_list = [self.my_parameters[i] for i in self.nq_layers_list]
            q_parameters_list = [self.my_parameters[i] for i in self.q_layers_list]
            return self.my_parameters, nn.ParameterList(nq_parameters_list), nn.ParameterList(q_parameters_list), self.factor_, self.b_, self.scale_, self.param_size_total
        else:
            return self.my_parameters

    def weight_clipper(self):
        for l in self.layers:
            l.weight_clipper()

    def train(self):
        for l in self.layers:
            l.train()

    def eval(self):
        for l in self.layers:
            l.eval()
