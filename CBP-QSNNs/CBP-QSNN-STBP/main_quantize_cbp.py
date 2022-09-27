from tqdm import tqdm
from collections import OrderedDict
import argparse
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable  # for lagrange multiplier
from functions import *
from q_module import *
from q_model import *

"""
### train ###
python main_quantize_cbp.py --dataset CIFAR10 --mode train --decay 0.25 --thresh 0.5 --lens 0.5 --T 8 --quant bin (okay)
python main_quantize_cbp.py --dataset CIFAR10 --mode train --decay 0.25 --thresh 0.5 --lens 0.5 --T 8 --quant ter (okay)

python main_quantize_cbp.py --dataset CIFAR100 --mode train --decay 0.8 --thresh 0.5 --lens 0.5 --T 8 --quant bin (okay)
python main_quantize_cbp.py --dataset CIFAR100 --mode train --decay 0.8 --thresh 0.5 --lens 0.5 --T 8 --quant ter (okay)

### eval ###
python main_quantize_cbp.py --dataset CIFAR10 --mode eval --decay 0.25 --thresh 0.5 --lens 0.5 --T 8 --quant bin (okay)
python main_quantize_cbp.py --dataset CIFAR10 --mode eval --decay 0.25 --thresh 0.5 --lens 0.5 --T 8 --quant ter (okay)

python main_quantize_cbp.py --dataset CIFAR100 --mode eval --decay 0.8 --thresh 0.5 --lens 0.5 --T 8 --quant bin (okay)
python main_quantize_cbp.py --dataset CIFAR100 --mode eval --decay 0.8 --thresh 0.5 --lens 0.5 --T 8 --quant ter (okay)

"""


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Pre-train')

parser.add_argument('--gpu', type=str, default="0", help='GPU id to use')
parser.add_argument('--dataset', type=str, default='CIFAR10', help='Which dataset to run (CIFAR10 or CIFAR100)')
parser.add_argument('--mode', type=str, default='train', help='Whether to train or eval')
parser.add_argument('--quant', type=str, default='bin', help='Weight quantization (bin or ter)')

parser.add_argument('--decay', type=float, default=0.25, help='Decay factor (wehn dt and tau_m is not defined.')
parser.add_argument('--thresh', type=float, default=0.5, help='Threshold voltage')
parser.add_argument('--lens', type=float, default=0.5, help='Hyper-parameter of surrogate gradient')
parser.add_argument('--T', type=int, default=8, help='Number of timesteps')

parser.add_argument('--num_workers', type=int, default=0, help='')
parser.add_argument('--batch_size', type=int, default=50, help='Batch size')
parser.add_argument('--num_epochs', type=int, default=100, help='Maximum number of epochs')
parser.add_argument('--opt', type=str, default='sgd', help='Optimizer(adam or sgd')
parser.add_argument('--learning_rate', type=float, default=5e-3, help='learning rate for weight update')
parser.add_argument('--learning_rate_lambda', type=float, default=5e-4, help='learning rate for multiplier update')

args = parser.parse_args()
print(args)


### Utils for applying CBP ###

def getparameters(model):
    lamb = []        # Lagrange multiplier
    qweight = []     # weight to be quantized
    nqweight = []    # weight not to be quantized
    otherparam = []  # otherparams such as batchnorm, bias, ...
    factor = []      # factors of each quantized layers
    b = []           # b of each quantized layers (median)
    scale = []       # scale factor of each quantized layers
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
                
    return lamb, qweight, nqweight, otherparam, factor, b, scale, param_size

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

def adjust_lr(optimizer, decrease_rate):
    for p in optimizer.param_groups:
        p['lr']*=decrease_rate 



def train(model, outs, train_dl, criterion, optimizer, qweight, lamb, scale, factor, b, ucs):
    lagsum = torch.zeros((1)).cuda()
    loss_train = torch.zeros((1)).cuda()     
    
    with tqdm(total=len(train_dl)) as pbar:
        for i, (images, labels) in enumerate(train_dl):
            model.zero_grad()
            optimizer.zero_grad() 
            
            images = images.to(device)
            outputs = model(images)
            labels_ = torch.zeros(args.batch_size, outs).scatter_(1, labels.view(-1, 1), 1)
                
            loss_network = criterion(outputs.cpu(), labels_)
            const = torch.zeros(1).cuda()
            for j in range(len(lamb)):
                const = const + constraints(qweight[j], lamb[j].detach(), scale[j], factor[j], b[j], ucs)               
            lag = loss_network + const
            lagsum += lag.detach() 
            loss_train += loss_network.detach()
        
            lag.backward(retain_graph=True)
            torch.nn.utils.clip_grad_value_(parameters=model.parameters(), clip_value=1)
            optimizer.step()
            for p in qweight:
                p.data.clamp_(min=-1, max=1)
                
            pbar.update(1)
            
    return loss_train, lagsum


def test(model, outs, test_dl, criterion):
    test_loss = 0
    correct = 0
    total = 0
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_dl):
            inputs = inputs.to(device)
            outputs = model(inputs)
            targets_ = torch.zeros(args.batch_size, outs).scatter_(1, targets.view(-1, 1), 1)
                
            loss = criterion(outputs.cpu(), targets_)
            _, predicted = outputs.cpu().max(1)

            test_loss += loss.item() / len(test_dl)
            total += float(targets.size(0))
            correct += float(predicted.eq(targets).sum().item())
            
    acc = 100. * float(correct) / float(total)
    
    return test_loss, acc
    

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    out_dir = './checkpoint'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    progress_dir = './progress'
    if not os.path.exists(progress_dir):
        os.makedirs(progress_dir)
    
    normalize_cifar = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          normalize_cifar])

    transform_test = transforms.Compose([transforms.ToTensor(),
                                         normalize_cifar])
    
    if args.dataset == 'CIFAR10':
        outs = 10
        train_dataset = torchvision.datasets.CIFAR10(root='./datasets', train=True, download=True, transform=transform_train)
        train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

        test_dataset = torchvision.datasets.CIFAR10(root='./datasets', train=False, download=True, transform=transform_test)
        test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        
    elif args.dataset == 'CIFAR100':
        outs = 100
        train_dataset = torchvision.datasets.CIFAR100(root='./datasets', train=True, download=True, transform=transform_train)
        train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

        test_dataset = torchvision.datasets.CIFAR100(root='./datasets', train=False, download=True, transform=transform_test)
        test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)      
    
    ### Define model ###
    model = Q_Conv_SNN(outs=outs, decay=args.decay, thresh=args.thresh, lens=args.lens, batch_size=args.batch_size, timesteps=args.T, mode=args.quant)
    model.to(device)
    
    ### Get pre-trained weight ###
    checkpoint_path = './trained_params/' + 'STBP_' + args.dataset + '_fp32_pretrained.pth' # from 'trained_params' directory
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    new_state_dict = OrderedDict()
    for n,v in checkpoint.items():
        new_state_dict[n] = v

    model_state_dict = model.state_dict()
    model_state_dict.update(new_state_dict)
    model.load_state_dict(model_state_dict)
    
    criterion = nn.MSELoss()
    
    if args.mode == 'train':
        
        ### Initialization of scale ###
        for p in model.modules():
            if isinstance(p, (QConv2d, QLinear)):
                p.scale.data[0] = p.weight.abs().mean()
            
        ### Get parameters ###
        lamb, qweight, nqweight, otherparam, factor, b, scale, param_size = getparameters(model)

        ### Optimizer ###
        optimizer = torch.optim.SGD([{'params':qweight, 'lr':args.learning_rate},
                                     {'params':nqweight, 'lr':args.learning_rate},
                                     {'params':otherparam, 'lr':args.learning_rate}], momentum=0.9)
        optimizer2 = torch.optim.Adam([{'params':lamb, 'lr':args.learning_rate_lambda}])
        
        ### Initialization of unconstrained window ###
        g = 1
        ucs = 1-1/g
        
        ### Initial update of multiplier ###
        updatelambda(optimizer2, qweight, lamb, scale, factor, b, ucs)
        
        ### Save epoch, lagsum_pre, test accuracy, cfs ###
        progress = np.zeros((1,4))
        
        ### lagsum_max, period ###
        lagsum_pre = 1e10  # lagsum_max in algorithm 1     
        period = 0;  period_max = 20;     
        
        best_acc = 0
        best_epoch = 0
        
        for epoch in range(args.num_epochs):
            _, lagsum = train(model, outs, train_dl, criterion, optimizer, qweight, lamb, scale, factor, b, ucs)
            period += 1
            print('Epoch {}/{} ... lagsum, lagsum_pre = {},{}'.format(epoch+1, args.num_epochs, lagsum.item(), lagsum_pre))
            
            if lagsum >= lagsum_pre or period == period_max:
                print(f'Lambda update start at {epoch} epoch')

                ### Update of unconstrained window ###
                if g < 10:
                    g += 1
                else:
                    g += 10
                ucs = 1-1/g 

                ### Learning rate scheduler ###
                if g == 20:
                    adjust_lr(optimizer, 0.1)

                ### Update of lambda ###
                updatelambda(optimizer2, qweight, lamb, scale, factor, b, ucs)

                ### Reset lagsum and period ###
                lagsum_pre = 1e10
                period = 0
                
            else:
                lagsum_pre = lagsum.item()
                
                
            ### Calculate test accuracy and cfs ###                   
            _, acc = test(model, outs, test_dl, criterion)
            cfs = CFS(qweight, param_size, scale, factor, b)
            print('Epoch {}/{} ... test_acc, cfs = {}, {}'.format(epoch+1, args.num_epochs, acc, cfs))            

            
            ### Save data ###
            progress=np.append(progress, np.array([[epoch, lagsum_pre, acc, cfs]]), axis=0)
            progress_data=pd.DataFrame(progress)
            progress_data.to_csv(progress_dir + '/' + args.dataset + '_cbp_' + args.quant + '_progress.txt', index=False, header=False,sep='\t')

            state = {'epoch': epoch,
                     'net': model.state_dict(),
                     'optimizer1': optimizer.state_dict(),
                     'optimizer2': optimizer2.state_dict(),
                     'lamb': lamb,
                     'period': period,
                     'ucs': ucs,
                     'g' : g,
                     'progress' : progress} 
            
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                torch.save(state, out_dir + '/' + args.dataset + '_cbp_' +  args.quant + f'_best.pth')
            else:
                torch.save(state, out_dir + '/' + args.dataset + '_cbp_' +  args.quant + f'.pth')
             
            print(f'Best Test Accuracy : {best_acc} at {best_epoch} Epoch ...')
    
    elif args.mode == 'eval':
        
        model_state_dict = torch.load('./trained_params/' + 'STBP_' + args.dataset + '_'+ args.quant + '_cbp_prequantized.pth')  # from 'trained_params' directory
        model.load_state_dict(model_state_dict)
        _, acc = test(model, outs, test_dl, criterion)
        print("Test Accuracy: {:.3f}".format(acc))
        

if __name__ == '__main__':
    main()