import datetime
import os
import time
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import sys
from torch.cuda import amp
import q_smodels
import argparse
from spikingjelly.clock_driven import functional
from spikingjelly.datasets import cifar10_dvs
import math
from tqdm import tqdm
from q_module import *
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode
from torch.autograd import Variable  # for lagrange multiplier

_seed_ = 2022
import random
random.seed(2022)

torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.cuda.manual_seed_all(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import numpy as np
np.random.seed(_seed_)

'''
### train ###
python main_quantize_cbp.py -amp -out_dir ./logs -model SEWResNet -cnf ADD -device cuda:0 -b 16 -opt SGD -lr 0.1 -lr_lambda 0.01 -epochs 64 -quant bin -period 20
python main_quantize_cbp.py -amp -out_dir ./logs -model SEWResNet -cnf ADD -device cuda:0 -b 16 -opt SGD -lr 0.1 -lr_lambda 0.01 -epochs 64 -quant ter -period 20

### test ###
python main_quantize_cbp.py -out_dir ./logs -model SEWResNet -cnf ADD -device cuda:0 -data_dir ./datasets/CIFAR10DVS -test-only -quant bin (73.0)
python main_quantize_cbp.py -out_dir ./logs -model SEWResNet -cnf ADD -device cuda:0 -data_dir ./datasets/CIFAR10DVS -test-only -quant ter (73.5)
'''

'''
training

python main_quantize_cbp.py -amp -out_dir ./logs -model SEWResNet -cnf ADD -device cuda:0 -b 16 -opt SGD -lr 0.1 -lr_lambda 0.01 -epochs 64 -quant bin -period 20  73.4  # seed 2020
python main_quantize_cbp.py -amp -out_dir ./logs -model SEWResNet -cnf ADD -device cuda:1 -b 16 -opt SGD -lr 0.1 -lr_lambda 0.01 -epochs 64 -quant ter -period 20  74.3  # seed 2020
python main_quantize_cbp.py -amp -out_dir ./logs -model SEWResNet -cnf ADD -device cuda:0 -b 16 -opt SGD -lr 0.1 -lr_lambda 0.01 -epochs 64 -quant bin -period 20  73.3  # seed 2021
python main_quantize_cbp.py -amp -out_dir ./logs -model SEWResNet -cnf ADD -device cuda:1 -b 16 -opt SGD -lr 0.1 -lr_lambda 0.01 -epochs 64 -quant ter -period 20  73.8  # seed 2021
python main_quantize_cbp.py -amp -out_dir ./logs -model SEWResNet -cnf ADD -device cuda:0 -b 16 -opt SGD -lr 0.1 -lr_lambda 0.01 -epochs 64 -quant bin -period 20  72.9  # seed 2022
python main_quantize_cbp.py -amp -out_dir ./logs -model SEWResNet -cnf ADD -device cuda:1 -b 16 -opt SGD -lr 0.1 -lr_lambda 0.01 -epochs 64 -quant ter -period 20  74.1  # seed 2022

python main_quantize_cbp.py -out_dir ./logs -model SEWResNet -b 50 -cnf ADD -device cuda:0 -data_dir ./datasets/CIFAR10DVS -test-only -quant bin
python main_quantize_cbp.py -out_dir ./logs -model SEWResNet -b 50 -cnf ADD -device cuda:0 -data_dir ./datasets/CIFAR10DVS -test-only -quant ter

'''

### Utils for applying CBP ###

def getparameters(model):
    lamb = []         # Lagrange multiplier
    qweight = []      # weight to be quantized
    nqweight = []     # weight not to be quantized such as first, last layer parameter
    otherparam = []   # otherparams such as batchnorm, bias
    timeconstant = [] # timeconstant parameter of each layer
    factor = []       # factors of each quantized layers
    b = []            # b of each quantized layers (median)
    scale = []        # scale factor of each quantized layers 
    param_size = 0    
    for p in model.modules():
        if isinstance(p, (QConv2d, QLinear)):
            qweight += [p.weight]
            lamb += [Variable(torch.full(p.weight.shape, 0).float().cuda(), requires_grad=True)]
            if p.bias != None:
                otherparam += [p.bias]
            scale += [p.scale]
            factor += [p.factor]
            b += [p.b]
            param_size += p.weight.numel()
        elif isinstance(p, (nn.Conv2d, nn.Linear)):
            nqweight += [p.weight]
            if p.bias != None:
                otherparam += [p.bias]
        elif isinstance(p, (nn.BatchNorm2d, nn.BatchNorm1d)):
            otherparam += [p.weight]
            otherparam += [p.bias]
        elif isinstance(p, (MultiStepParametricLIFNode)):
            timeconstant += [p.w]
    return lamb, qweight, nqweight, otherparam, timeconstant, factor, b, scale, param_size

def updatelambda(optimizer2, qweight, lamb, scale, factor, b, ucs):
    const = torch.zeros(1).cuda()
    for i in range(len(lamb)):
        const = const + constraints(qweight[i].detach(), lamb[i], scale[i], factor[i], b[i], ucs)
    optimizer2.zero_grad()
    (-const).backward(retain_graph=True) # gradient ascent
    optimizer2.step()
    
def constraints(weight, lamb, scale, factor, b, ucs):
    out = constraint().apply(weight, scale, factor, b, ucs)
    return (out*lamb).sum()
    
def CFS(weight, size, scale, factor, b):
    cfstotal = 0
    for p, q, r, s in zip(weight, scale, factor, b):
        cfs = constraint().apply(p, q, r, s, 1) # set ucs to 1
        cfstotal += cfs.sum()
    return cfstotal.item()/size

def adjust_lr(optimizer, decrease_rate):
    for p in optimizer.param_groups:
        p['lr']*=decrease_rate


def split_to_train_test_set(train_ratio: float, origin_dataset: torch.utils.data.Dataset, num_classes: int, random_split: bool = False):
    '''
    :param train_ratio: split the ratio of the origin dataset as the train set
    :type train_ratio: float
    :param origin_dataset: the origin dataset
    :type origin_dataset: torch.utils.data.Dataset
    :param num_classes: total classes number, e.g., ``10`` for the MNIST dataset
    :type num_classes: int
    :param random_split: If ``False``, the front ratio of samples in each classes will
            be included in train set, while the reset will be included in test set.
            If ``True``, this function will split samples in each classes randomly. The randomness is controlled by
            ``numpy.randon.seed``
    :type random_split: int
    :return: a tuple ``(train_set, test_set)``
    :rtype: tuple
    '''
    label_idx = []
    for i in range(num_classes):
        label_idx.append([])

    for i, item in enumerate(origin_dataset):
        y = item[1]
        if isinstance(y, np.ndarray) or isinstance(y, torch.Tensor):
            y = y.item()
        label_idx[y].append(i)
    train_idx = []
    test_idx = []
    if random_split:
        for i in range(num_classes):
            np.random.shuffle(label_idx[i])

    for i in range(num_classes):
        pos = math.ceil(label_idx[i].__len__() * train_ratio)
        train_idx.extend(label_idx[i][0: pos])
        test_idx.extend(label_idx[i][pos: label_idx[i].__len__()])

    return torch.utils.data.Subset(origin_dataset, train_idx), torch.utils.data.Subset(origin_dataset, test_idx)

def main():

    parser = argparse.ArgumentParser(description='PyTorch Classification Training')
    parser.add_argument('-T', default=16, type=int, help='simulating time-steps')
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-b', default=16, type=int, help='batch size')
    parser.add_argument('-epochs', default=64, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-data_dir', type=str, default='./datasets/CIFAR10DVS')
    parser.add_argument('-out_dir', type=str, help='root dir for saving logs and checkpoint', default='./logs')

    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument("-test-only", dest="test_only", help="only test the model", action="store_true")
    parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')

    parser.add_argument('-opt', type=str, help='use which optimizer. SDG or Adam', default='SGD')
    parser.add_argument('-lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('-lr_lambda', default=0.01, type=float, help='learning rate for lambda update')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-model', type=str)
    parser.add_argument('-cnf', type=str)
    
    parser.add_argument('-quant', default='bin', type=str, help='quantization')              
    parser.add_argument('-period', type=int, help='max period of lambda update', default=20)                      
    
    parser.add_argument('-T_train', default=None, type=int)
    parser.add_argument('-dts_cache', type=str, default='./dts_cache')

    args = parser.parse_args()
    #print(args)
    
    ### Define model ###
    torch.cuda.set_device(args.device) 
    net = q_smodels.__dict__[args.model](args.cnf, args.quant) 
    #print(net)
    print("Creating model")
    net.to(args.device)
    
    ### Get pre-trained weight
    checkpoint_path = './trained_params/SEW_ResNet_CIFAR10DVS_fp32_pretrained.pth'
    checkpoint = torch.load(checkpoint_path)
    net_state_dict = net.state_dict()
    net_state_dict.update(checkpoint)
    net.load_state_dict(net_state_dict)
    print("Loading model")
    
    ### Initialization of scale ###
    for p in net.modules():
        if isinstance(p, (QConv2d, QLinear)):
            p.scale.data[0] = p.weight.abs().mean()
    
    ### Get parameters ###
    lamb, qweight, nqweight, otherparam, timeconstant, factor, b, scale, param_size = getparameters(net)
    
    ### Optimizer ### 
    optimizer = None
    if args.opt == 'SGD':
        optimizer1 = torch.optim.SGD([{'params':qweight, 'lr':args.lr},
                                      {'params':nqweight, 'lr':args.lr},
                                      {'params':otherparam, 'lr':args.lr},
                                      {'params':timeconstant, 'lr':args.lr}], momentum=0.9) 
        optimizer2 = torch.optim.Adam([{'params':lamb, 'lr':args.lr_lambda}])
    elif args.opt == 'Adam':
        optimizer1 = torch.optim.Adam([{'params':qweight, 'lr':args.lr},
                                       {'params':nqweight, 'lr':args.lr},
                                       {'params':otherparam, 'lr':args.lr},
                                       {'params':timeconstant, 'lr':args.lr}])
        optimizer2 = torch.optim.Adam([{'params':lamb, 'lr':args.lr_lambda}])
    else:
        raise NotImplementedError(args.opt)

    train_set_pth = os.path.join(args.dts_cache, f'train_set_{args.T}.pt')
    test_set_pth = os.path.join(args.dts_cache, f'test_set_{args.T}.pt')
    if os.path.exists(train_set_pth) and os.path.exists(test_set_pth):
        train_set = torch.load(train_set_pth)
        test_set = torch.load(test_set_pth)
    else:
        origin_set = cifar10_dvs.CIFAR10DVS(root=args.data_dir, data_type='frame', frames_number=args.T, split_by='number')

        train_set, test_set = split_to_train_test_set(0.9, origin_set, 10)
        if not os.path.exists(args.dts_cache):
            os.makedirs(args.dts_cache)
        torch.save(train_set, train_set_pth)
        torch.save(test_set, test_set_pth)

    train_data_loader = DataLoader(
        dataset=train_set,
        batch_size=args.b,
        shuffle=True,
        num_workers=args.j,
        drop_last=True,
        pin_memory=True)

    test_data_loader = DataLoader(
        dataset=test_set,
        batch_size=args.b,
        shuffle=False,
        num_workers=args.j,
        drop_last=False,
        pin_memory=True)

    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    start_epoch = 0
    max_test_acc = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        optimizer1.load_state_dict(checkpoint['optimizer1'])
        optimizer2.load_state_dict(checkpoint['optimizer2'])
        start_epoch = checkpoint['epoch'] + 1
        lamb = checkpoint['lamb']
        lagsum_pre = checkpoint['lagsum_pre']
        period = checkpoint['period']
        ucs = checkpoint['ucs']
        g = checkpoint['g']
        progress = checkpoint['progress']
        max_test_acc = checkpoint['max_test_acc']
        
    if args.test_only:
        net_state_dict = torch.load('./trained_params/SEW_ResNet_CIFAR10DVS_' + args.quant + '_cbp_prequantized.pth')  # from 'trained_params' directory
        net.load_state_dict(net_state_dict)
        
        ### evaluate ###
        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for frame, label in test_data_loader:
                frame = frame.float().to(args.device)
                label = label.to(args.device)
                out_fr = net(frame)
                loss = F.cross_entropy(out_fr, label)

                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(net)

        test_loss /= test_samples
        test_acc /= test_samples
        acc = test_acc * 100.
        
        print("Test Loss: {:.5f}.. ".format(test_loss), "Test Accuracy: {:.3f}".format(acc))
        
        return
        
    out_dir = os.path.join(args.out_dir, f'{args.model}_{args.cnf}_T_{args.T}_T_train_{args.T_train}_b_{args.b}_{args.opt}_lr_{args.lr}_lrlambda_{args.lr_lambda}')

    if args.amp:
        out_dir += '_amp'
    
    out_dir += f'_period_{args.period}_{args.quant}_3'  # added
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Mkdir {out_dir}.')
    else:
        print(out_dir)
        assert args.resume is not None

    pt_dir = out_dir + '_pt'
    if not os.path.exists(pt_dir):
        os.makedirs(pt_dir)
        print(f'Mkdir {pt_dir}.')

    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))

    writer = SummaryWriter(os.path.join(out_dir, 'logs'), purge_step=start_epoch)
    
    if args.resume:
        pass
    else:
        ### Initialization of unconstrained window ###
        g = 1
        ucs = 1-1/g
        
        ### Initial update of multiplier ###
        updatelambda(optimizer2, qweight, lamb, scale, factor, b, ucs)
        
        ### Save epoch, train acc, test acc, train loss, test loss, cfs, lagsum_pre ###
        progress = np.zeros((1,7))
        
        ### lagsum_max, period ###
        lagsum_pre = 1e10  # lagsum_max in algorithm 1
        period = 0
    
    print("Start training")
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        
        lagsum = torch.zeros(1).to(args.device) 
        
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        
        ### train one epoch ###
        with tqdm(total=len(train_data_loader)) as pbar:
            for frame, label in train_data_loader:
                
                optimizer1.zero_grad()
                frame = frame.float().to(args.device)

                if args.T_train:
                    sec_list = np.random.choice(frame.shape[1], args.T_train, replace=False)
                    sec_list.sort()
                    frame = frame[:, sec_list]

                label = label.to(args.device)
                if args.amp:
                    with amp.autocast():
                        out_fr = net(frame)
                        loss_network = F.cross_entropy(out_fr, label)
                        const = torch.zeros(1).to(args.device)
                        for i in range(len(lamb)):
                            const = const + constraints(qweight[i], lamb[i].detach(), scale[i], factor[i], b[i], ucs)
                        lag = loss_network + const
                        lagsum += lag.detach()
                        
                    scaler.scale(lag).backward(retain_graph=True) 
                    torch.nn.utils.clip_grad_value_(parameters=net.parameters(), clip_value=1) 
                    scaler.step(optimizer1)
                    scaler.update()
                    for p in qweight:
                        p.data.clamp_(min=-1, max=1)
                else:
                    out_fr = net(frame)
                    loss_network = F.cross_entropy(out_fr, label)
                    const = torch.zeros(1).to(args.device)
                    for i in range(len(lamb)):
                        const = const + constraints(qweight[i], lamb[i].detach(), scale[i], factor[i], b[i], ucs)
                    lag = loss_network + const
                    lagsum += lag.detach()
                            
                    lag.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_value_(parameters=net.parameters(), clip_value=1)
                    optimizer1.step()
                    for p in qweight:
                        p.data.clamp_(min=-1, max=1)

                train_samples += label.numel()
                train_loss += loss_network.item() * label.numel()
                train_acc += (out_fr.argmax(1) == label).float().sum().item()

                functional.reset_net(net)
                pbar.update(1)
                
        period += 1
                    
        train_loss /= train_samples
        train_acc /= train_samples

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)
        
        print(f'epoch={epoch}...lagsum_pre={lagsum_pre}...lagsum={lagsum}') 
        if lagsum >= lagsum_pre or period == args.period:
            print("Lambda update")
            
            ### Update of unconstrained window ###
            if g<10:
                g+=1
            else:
                g+=10
            ucs = 1-1/g
            
            ### Learning rate scheduler ###
            if g==20:
                adjust_lr(optimizer1, 0.1)
            
            ### Update of lambda ###
            updatelambda(optimizer2, qweight, lamb, scale, factor, b, ucs)
            
            ### Reset lagsum and period ###
            lagsum_pre = 1e10
            period = 0
        else:
            lagsum_pre = lagsum.item()

        ### evaluate one epoch ###
        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for frame, label in test_data_loader:
                frame = frame.float().to(args.device)
                label = label.to(args.device)
                out_fr = net(frame)
                loss = F.cross_entropy(out_fr, label)

                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(net)

        test_loss /= test_samples
        test_acc /= test_samples
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)
        
        ### calculate CFS ###
        cfs=CFS(qweight,param_size, scale,factor, b)
        
        ### Save data ###
        progress=np.append(progress,np.array([[epoch, train_acc, test_acc, train_loss, test_loss, cfs, lagsum_pre]]), axis=0)
        progress_data=pd.DataFrame(progress)
        progress_data.to_csv(out_dir+"/progress.txt",
        index=False, header=False,sep='\t')
        #####################################

        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True

        checkpoint = {
            'net': net.state_dict(),
            'factor' : factor,
            'optimizer1': optimizer1.state_dict(), 
            'optimizer2': optimizer2.state_dict(), 
            'epoch': epoch,
            'lamb' : lamb,                         
            'lagsum_pre' : lagsum_pre,             
            'period' : period,                     
            'ucs' : ucs,
            'g' : g,
            'progress' : progress,
            'args' : args,
            'max_test_acc': max_test_acc
        }

        if save_max:
            torch.save(checkpoint, os.path.join(pt_dir, 'checkpoint_max.pth'))
        
        if epoch == 0 or (epoch+1) % 10 == 0: 
            torch.save(checkpoint, os.path.join(pt_dir, f'checkpoint_{epoch}.pth'))

        
        for item in sys.argv:
            print(item, end=' ')
        print('')
        print(args)
        print(out_dir)
        total_time = time.time() - start_time
        print(f'epoch={epoch}, train_loss={train_loss}, train_acc={train_acc}, test_loss={test_loss}, test_acc={test_acc}, max_test_acc={max_test_acc}, total_time={total_time}, escape_time={(datetime.datetime.now()+datetime.timedelta(seconds=total_time * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}')

if __name__ == '__main__':
    main()
    

