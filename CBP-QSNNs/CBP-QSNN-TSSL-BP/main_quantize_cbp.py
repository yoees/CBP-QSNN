import os
import sys
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from network_parser import parse
from datasets import loadCIFAR10
import logging
import cnns
from utils import learningStats
from utils import aboutCudaDevices
from utils import EarlyStopping
import functions.loss_f as loss_f
import pycuda.driver as cuda
from torch.nn.utils import clip_grad_norm_
import global_v as glv
from layers.q_module import *
from torch.autograd import Variable # for lagrange multiplier

max_accuracy = 0
min_loss = 1000


### Utils for applying CBP ###

def getparameters(network):
    lamb = []  # Lagrange multiplier
    #  _, weight not to be quantized, weight to be quantized, factors of each quantized layers, b of each quantized layers (median), scale factor of each quantized layers, param_size 
    _, nqweight, qweight, factor, b, scale, param_size = network.get_parameters()  
    for p in qweight:
        lamb += [Variable(torch.full(p.shape, 0).float().cuda(), requires_grad=True)]
    return lamb, nqweight, qweight, factor, b, scale, param_size

def constraints(weight, lamb, scale, factor, b, ucs):
    out = constraint().apply(weight, scale, factor, b, ucs)
    return (out*lamb).sum()

def updatelambda(optimizer2, qweight, lamb, scale, factor, b, ucs):
    const = torch.zeros(1).cuda()
    for i in range(len(lamb)):
        const = const + constraints(qweight[i].detach(), lamb[i], scale[i], factor[i], b[i], ucs)
    optimizer2.zero_grad()
    (-const).backward(retain_graph=True) # gradient ascent
    optimizer2.step()
    
def CFS(weight, size, scale, factor, b):
    cfstotal = 0
    for p, q, r, s in zip(weight, scale, factor, b):
        cfs = constraint().apply(p, q, r, s, 1) # set ucs to 1
        cfstotal += cfs.sum()
    return cfstotal.item()/size



def train(network, trainloader, opti, epoch, states, network_config, layers_config, err, qweight, lamb, scale, factor, b, ucs):
    network.train()
    global max_accuracy
    global min_loss
    logging.info('\nEpoch: %d', epoch)
    train_loss = 0
    correct = 0
    total = 0
    n_steps = network_config['n_steps']
    n_class = network_config['n_class']
    batch_size = network_config['batch_size']
    time = datetime.now()
    
    lagsum = torch.zeros((1)).cuda()

    if network_config['loss'] == "kernel":
        # set target signal
        if n_steps >= 10:
            desired_spikes = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]).repeat(int(n_steps/10))
        else:
            desired_spikes = torch.tensor([0, 1, 1, 1, 1]).repeat(int(n_steps/5))
        desired_spikes = desired_spikes.view(1, 1, 1, 1, n_steps).cuda()
        desired_spikes = loss_f.psp(desired_spikes, network_config).view(1, 1, 1, n_steps)
        
    des_str = "Training @ epoch " + str(epoch)
    for batch_idx, (inputs, labels) in enumerate(trainloader):
        start_time = datetime.now()
        targets = torch.zeros(labels.shape[0], n_class, 1, 1, n_steps).cuda() 
        if network_config["rule"] == "TSSLBP":
            if len(inputs.shape) < 5:
                inputs = inputs.unsqueeze_(-1).repeat(1, 1, 1, 1, n_steps)
            # forward pass
            labels = labels.cuda()
            inputs = inputs.cuda()
            inputs.type(torch.float32)
            outputs = network.forward(inputs, epoch, True)

            if network_config['loss'] == "count":
                # set target signal
                desired_count = network_config['desired_count']
                undesired_count = network_config['undesired_count']

                targets = torch.ones(outputs.shape[0], outputs.shape[1], 1, 1).cuda() * undesired_count
                for i in range(len(labels)):
                    targets[i, labels[i], ...] = desired_count
                loss = err.spike_count(outputs, targets, network_config, layers_config[list(layers_config.keys())[-1]])
            elif network_config['loss'] == "kernel":
                targets.zero_()
                for i in range(len(labels)):
                    targets[i, labels[i], ...] = desired_spikes
                loss = err.spike_kernel(outputs, targets, network_config)
            elif network_config['loss'] == "softmax":
                # set target signal
                loss = err.spike_soft_max(outputs, labels)
            else:
                raise Exception('Unrecognized loss function.')
            
            const = torch.zeros(1).cuda()
            for j in range(len(lamb)):
                const = const + constraints(qweight[j], lamb[j].detach(), scale[j], factor[j], b[j], ucs)
            lag = loss + const
            lagsum += lag.detach()

            # backward pass
            opti.zero_grad()
            lag.backward(retain_graph=True)
            clip_grad_norm_(network.get_parameters()[0], 1) 
            opti.step()
            network.weight_clipper()
            for p in qweight:
                p.data.clamp_(min=-1, max=1)

            spike_counts = torch.sum(outputs, dim=4).squeeze_(-1).squeeze_(-1).detach().cpu().numpy()
            predicted = np.argmax(spike_counts, axis=1)
            train_loss += torch.sum(loss).item()
            labels = labels.cpu().numpy()
            total += len(labels)
            correct += (predicted == labels).sum().item()
        else:
            raise Exception('Unrecognized rule name.')

        states.training.correctSamples = correct
        states.training.numSamples = total
        states.training.lossSum += loss.cpu().data.item() 
        states.print(epoch, batch_idx, (datetime.now() - time).total_seconds())

    total_accuracy = correct / total
    total_loss = train_loss / total
    if total_accuracy > max_accuracy:
        max_accuracy = total_accuracy
    if min_loss > total_loss:
        min_loss = total_loss

    logging.info("Train Accuracy: %.3f (%.3f). Loss: %.3f (%.3f)\n", 100. * total_accuracy, 100 * max_accuracy, total_loss, min_loss)
    
    return lagsum


def test(network, testloader, epoch, states, network_config, layers_config, early_stopping, mode):
    network.eval()
    global best_acc
    global best_epoch
    global save_name
    correct = 0
    total = 0
    n_steps = network_config['n_steps']
    n_class = network_config['n_class']
    time = datetime.now()
    y_pred = []
    y_true = []
    des_str = "Testing @ epoch " + str(epoch)
    for batch_idx, (inputs, labels) in enumerate(testloader):
        if network_config["rule"] == "TSSLBP":
            if len(inputs.shape) < 5:
                inputs = inputs.unsqueeze_(-1).repeat(1, 1, 1, 1, n_steps)
            # forward pass
            labels = labels.cuda()
            inputs = inputs.cuda()
            outputs = network.forward(inputs, epoch, False)

            spike_counts = torch.sum(outputs, dim=4).squeeze_(-1).squeeze_(-1).detach().cpu().numpy()
            predicted = np.argmax(spike_counts, axis=1)
            labels = labels.cpu().numpy()
            y_pred.append(predicted)
            y_true.append(labels)
            total += len(labels)
            correct += (predicted == labels).sum().item()
        else:
            raise Exception('Unrecognized rule name.')

        states.testing.correctSamples += (predicted == labels).sum().item()
        states.testing.numSamples = total
        states.print(epoch, batch_idx, (datetime.now() - time).total_seconds())

    test_accuracy = correct / total
    if test_accuracy > best_acc:
        best_epoch = epoch
        best_acc = test_accuracy

    logging.info("Test Accuracy: %.3f (%.3f).\n", 100. * test_accuracy, 100 * best_acc)
    
    # Save checkpoint.
    acc = 100. * correct / total
    if mode == 'train':
        early_stopping(acc, network, epoch, save_name)
    # Evaluate model
    elif mode == 'eval':
        print("Test Accuracy: ", acc)
    
    return acc
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', action='store', dest='config', help='The path of config file')
    parser.add_argument('-checkpoint', action='store', dest='checkpoint', help='The path of checkpoint, if use checkpoint')
    parser.add_argument('-mode', type=str, default='train', help='Whether to train or eval')
    parser.add_argument('-gpu', type=int, default=0, help='GPU device to use (default: 0)')
    parser.add_argument('-seed', type=int, default=3, help='random seed (default: 3)')
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        exit(0)

    if args.config is None:
        raise Exception('Unrecognized config file.')
    else:
        config_path = args.config
    
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    if not os.path.exists('./progress'):
        os.makedirs('./progress')
 
    dataset, quant = config_path.split('/')[-1].split('.')[0].split('_')
    save_name = dataset + '_' + quant
    logging.basicConfig(filename='./logs/' + save_name + '_' + args.mode + '.log', level=logging.INFO)   
    logging.info("start parsing settings")
    params = parse(config_path)    
    logging.info("finish parsing settings")
    
    # check GPU
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    
    # set GPU
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    cudnn.enabled = True
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    
    glv.init(params['Network']['n_steps'], params['Network']['tau_s'] )
    
    logging.info("dataset loaded")
    if params['Network']['dataset'] == "CIFAR10":
        data_path = os.path.expanduser(params['Network']['data_path'])
        train_loader, test_loader = loadCIFAR10.get_cifar10(data_path, params['Network'])
    else:
        raise Exception('Unrecognized dataset name.')
    logging.info("dataset loaded")
    
    ### Define network ###
    net = cnns.Network(params['Network'], params['Layers'], list(train_loader.dataset[0][0].shape)).cuda()
    
    ### Get pre-trained weight ###
    checkpoint_path = './trained_params/' + 'TSSL_BP_' + dataset + '_fp32_pretrained.pth' # from 'trained_params' directory
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    net.load_state_dict(checkpoint)
    
    ### Initialization of scale ###
    for p in net.layers:
        if (p.__class__.__name__ == 'QConvLayer') or (p.__class__.__name__ == 'QLinearLayer'):
            p.scale.data[0] = p.weight.abs().mean()
    
    error = loss_f.SpikeLoss(params['Network']).cuda()

    best_acc = 0
    best_epoch = 0


    if args.mode == 'train':
        
        ### Get parameters ###
        lamb, nqweight, qweight, factor, b, scale, param_size = getparameters(net)

        ### Optimizer ###
        optimizer = torch.optim.AdamW([{'params':qweight, 'lr':params['Network']['lr']}])
        optimizer2 = torch.optim.AdamW([{'params': lamb, 'lr':params['Network']['lr_lambda']}])

        ### Initialization of unconstrained window ###
        g = 1
        ucs = 1-1/g

        ### Initial update of multiplier ###
        updatelambda(optimizer2, qweight, lamb, scale, factor, b, ucs)

        ### Save epoch, test acc, cfs ###
        progress = np.zeros((1,3))

        ### lagsum_max, period ###
        lagsum_pre = 1e10  # lagsum_max in algorithm 1  
        period = 0; period_max = 20;

        l_states = learningStats()
        early_stopping = EarlyStopping()

        for e in range(params['Network']['epochs']):
            # training...
            l_states.training.reset()
            lagsum = train(net, train_loader, optimizer, e, l_states, params['Network'], params['Layers'], error,
                           qweight, lamb, scale, factor, b, ucs)
            period += 1
            logging.info('Epoch {}... // lagsum, lagsum_pre = {}, {}'.format(e, lagsum.item(), lagsum_pre))
            l_states.training.update()
            if lagsum >= lagsum_pre or period == period_max:
                logging.info('lambda update ....')

                ### Update of unconstrained window ###
                if g < 10:
                    g += 1
                else:
                    g += 10
                ucs = 1-1/g

                ### Update of lambda ###
                updatelambda(optimizer2, qweight, lamb, scale, factor, b, ucs)

                ### Reset lagsum and period ###
                lagsum_pre = 1e10
                period = 0

            else:
                lagsum_pre = lagsum.item()

            ### Calculate test accuracy and cfs ### 
            l_states.testing.reset()
            test_acc = test(net, test_loader, e, l_states, params['Network'], params['Layers'], early_stopping, args.mode)
            cfs = CFS(qweight, param_size, scale, factor, b)
            l_states.testing.update()
            # if early_stopping.early_stop:
            #     break


            ### Save data ###
            progress=np.append(progress,np.array([[e, test_acc, cfs]]),axis=0)
            progress_data=pd.DataFrame(progress)
            progress_data.to_csv("./progress/{}_{}.txt".format(params['Network']['dataset'], params['Network']['mode']), 
                                 index=False, header=False,sep='\t')   # mode, trial  

            if e % 40 == 0: ## added
                torch.save({'model_state_dict':net.state_dict(),
                        'scale': scale,
                        'epoch':e,
                        'lamb':lamb,
                        'lagsum_pre':lagsum_pre,
                        'period':period,
                        'ucs':ucs,
                        'g':g,
                        },"./progress/{}_{}".format(params['Network']['dataset'], params['Network']['mode']) + "_%d.pth"%(e))  

        logging.info("Best Accuracy: %.3f, at epoch: %d \n", best_acc, best_epoch)
        
    if args.mode == 'eval':
        
        net_state_dict = torch.load('./trained_params/' + 'TSSL_BP_' + params['Network']['dataset'] + '_' + params['Network']['mode'] + '_cbp_prequantized.pth') # from 'trained_params' directory
        net.load_state_dict(net_state_dict)
        
        l_states = learningStats()
        early_stopping = EarlyStopping()
        
        e = 0 
        test(net, test_loader, e, l_states, params['Network'], params['Layers'], early_stopping, args.mode)

