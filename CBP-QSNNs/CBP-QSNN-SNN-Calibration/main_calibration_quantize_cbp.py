import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import time
import datetime
import os
import random
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

from main_train_ann import build_data
from CIFAR.models.vgg import VGG
from CIFAR.models.calibration import GetLayerInputOutput, bias_corr_model, weights_cali_model
from CIFAR.models.fold_bn import search_fold_and_remove_bn
from CIFAR.models.spiking_layer import SpikeModule, SpikeModel, get_maximum_activation

from CIFAR.models.q_vgg import VGG_q
from CIFAR.models.q_spiking_layer import SpikeModule_nq, SpikeModule_q, SpikeModel_q
from CIFAR.models.q_module import *
from torch.autograd import Variable  # for lagrange multiplier

def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    
"""
python main_calibration_quantize_cbp.py --dataset CIFAR10 --arch VGG16 --T 32 --calib light --dpath ./datasets --device cuda:0 --opt SGD --quant bin --lr 1e-2 --lr_lambda 1e-3 

python main_calibration_quantize_cbp.py --dataset CIFAR10 --arch VGG16 --T 32 --calib light --dpath ./datasets --device cuda:0 --opt SGD --quant ter --lr 1e-2 --lr_lambda 1e-3

python main_calibration_quantize_cbp.py --dataset CIFAR100 --arch VGG16 --T 32 --calib light --dpath ./datasets --device cuda:0 --opt SGD --quant bin --lr 1e-2 --lr_lambda 1e-3 

python main_calibration_quantize_cbp.py --dataset CIFAR100 --arch VGG16 --T 32 --calib light --dpath ./datasets --device cuda:0 --opt SGD --quant ter --lr 1e-2 --lr_lambda 1e-3

"""

@torch.no_grad()
def validate_model(test_loader, model):
    correct = 0
    total = 0
    model.eval()
    device = next(model.parameters()).device
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = outputs.cpu().max(1)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets).sum().item())
    print('Test Accuracy of the model on the 10000 test images: %.3f' % (100 * correct / total))
    return 100 * correct / total 

### Utils for applying CBP ###

def getparameters(model):
    lamb = []         # Lagrange multiplier
    qweight = []      # weight to be quantized
    nqweight = []     # weight not to be quantized such as first, last layer parameters
    otherparam = []   # otherparams such as bias
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
            # setting : not optimize avgpoolconvs' weight
            if p.weight.shape[-1] == 2:
                pass
            else:
                nqweight += [p.weight]
                if p.bias != None:
                    otherparam += [p.bias]
    return lamb, qweight, nqweight, otherparam, factor, b, scale, param_size

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


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', default='CIFAR10', type=str, help='dataset name', choices=['CIFAR10', 'CIFAR100'])
    parser.add_argument('--arch', default='VGG16', type=str, help='network architecture', choices=['VGG16'])
    parser.add_argument('--dpath', required=True, type=str, help='dataset directory')
    parser.add_argument('--seed', default=1000, type=int, help='random seed to reproduce results')
    parser.add_argument('--batch_size', default=128, type=int, help='minibatch size')

    parser.add_argument('--calib', default='none', type=str, help='calibration methods',
                        choices=['none', 'light', 'advanced'])
    parser.add_argument('--T', default=32, type=int, help='snn simulation length')
    parser.add_argument('--usebn', action='store_true', help='use batch normalization in ann') 
    parser.add_argument('--device', default='cuda', help='device')  ## for assign single gpu id
    
    ## -- for quantization -- ##
    parser.add_argument('-epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--opt', type=str, help='use which optimizer. SGD or Adam', default='SGD')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('--wd', default=5e-4, type=float, help='weight decay (L2 regularization)')
    parser.add_argument('--lr_lambda', default=1e-5, type=float, help='learning rate for lambda update')  
    
    parser.add_argument('--quant', default='bin', type=str, help='quantization')              
    parser.add_argument('--period', type=int, help='max period of lambda update', default=20)
    ## ----------------------- ##
    
    args = parser.parse_args()
    use_bn = args.usebn

    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    if not os.path.exists('./progress'):
        os.makedirs('./progress')
        
    # run the experiments for 3 times, with different random seeds
    for i in range(3): 
        trial = i
        
        #### DNN-to-SNN conversion (SNN-Calibration) ###
        model_save_name_cali = 'logs/' + args.arch + '_' + args.dataset + '_' + 'T' + str(args.T) + '_' + str(i) + '_' + args.calib +'_ckpt.pth' 
        
        seed_all(seed=args.seed + i)
        sim_length = args.T
        
        use_cifar10 = args.dataset == 'CIFAR10'
        
        train_loader, test_loader = build_data(dpath=args.dpath, cutout=True, use_cifar10=use_cifar10, auto_aug=True)
        
        if args.arch == 'VGG16':
            ann = VGG('VGG16', use_bn=use_bn, num_class=10 if use_cifar10 else 100)
        else:
            raise NotImplementedError
            
        args.wd = 5e-4 if use_bn else 1e-4
        bn_name = 'wBN' if use_bn else 'woBN' 
        load_path = 'trained_params/' + 'ANN_' + args.arch + '_' + bn_name + '_' + args.dataset + '_ckpt.pth'
        state_dict = torch.load(load_path, map_location='cpu')
        ann.load_state_dict(state_dict, strict=True)
        search_fold_and_remove_bn(ann)
        ann.to(args.device)
        
        snn = SpikeModel(model=ann, sim_length=sim_length)
        snn.to(args.device)
        
        mse = False if args.calib == 'none' else True 
        get_maximum_activation(train_loader, model=snn, momentum=0.9, iters=5, mse=mse, percentile=None,
                              sim_length=sim_length, channel_wise=False) # set channel_wise=False
        
        if args.calib == 'light': # we used light-pipline
            bias_corr_model(model=snn, train_loader=train_loader, correct_mempot=False)
        if args.calib == 'advanced':
            weights_cali_model(model=snn, train_loader=train_loader, num_cali_samples=1024, learning_rate=1e-5)
            bias_corr_model(model=snn, train_loader=train_loader, correct_mempot=True)
            
        snn_test_acc = validate_model(test_loader, snn)

        #### Rescale weight, bias, threshold ###
        if args.arch == 'VGG16':
            prev_thresh = 0
            count = 0
            for name, module in snn.named_modules():
                if isinstance(module, SpikeModule):
                    if count == 0:
                        if module.bias is not None:
                            module.weight.data = module.weight.data / module.threshold
                            module.bias.data = module.bias.data / module.threshold
                            prev_thresh = module.threshold
                            module.threshold = 1.
                            count += 1
                        else:
                            module.weight.data = module.weight.data * prev_thresh / module.threshold
                            prev_thresh = module.threshold
                            module.threshold = 1.
                    else:
                        if module.bias is not None:
                            module.weight.data = module.weight.data * prev_thresh / module.threshold
                            module.bias.data = module.bias.data / module.threshold
                            prev_thresh = module.threshold
                            module.threshold = 1.
                        else:
                            module.weight.data = module.weight.data * prev_thresh / module.threshold
                            prev_thresh = module.threshold
                            module.threshold = 1.
                    
        snn.set_spike_state(use_spike=True)
        
        threshold_dict = {}
        mem_pot_init_dict = {}
        for name, module in snn.named_modules():
            if isinstance(module, SpikeModule):
                threshold_dict[name] = module.threshold
                mem_pot_init_dict[name] = module.mem_pot_init
                
        checkpoint = {
            'net_snn': snn.state_dict(),
            'threshold_dict': threshold_dict,
            'mem_pot_init_dict': mem_pot_init_dict,
            'snn_test_acc': snn_test_acc
        }
        
        torch.save(checkpoint, model_save_name_cali)
        print("{} trial: DNN-to-SNN coversion end!!".format(i))

        
        ### Quantization ###
        model_save_name_quant = './logs/' + args.arch + '_' + args.dataset + '_' + 'T' + str(args.T) + '_' + args.opt + '_' \
                                'lr' + str(args.lr) + 'lr_lambda' + str(args.lr_lambda) + 'wd' + str(args.wd) + '_' + args.quant + '_' + str(i) +'_ckpt.pth'
        
        if args.arch == 'VGG16':
            ann_q = VGG_q('VGG16', use_bn=use_bn, num_class=10 if use_cifar10 else 100, mode=args.quant)
        else:
            raise NotImplementedError
            
        ### Define model ### 
        snn_q = SpikeModel_q(model=ann_q, sim_length=sim_length)
        
        ### Get pre-trained weight ###
        model_state_dict = snn_q.state_dict()      
        load = torch.load(model_save_name_cali, map_location='cpu')
        dict_ = load['net_snn'].copy()
        dict_revise = {}
        for key, value in dict_.items():
            key_list = key.split('.')
            if key_list[-2] != 'fc' and key_list[-2] != 'classifier' and key_list[-3] != 'downsample':
                revise_key = '.'.join(key_list[:-1] + ['conv'] + [key_list[-1]])
                dict_revise[revise_key] = value
            else:
                dict_revise[key] = value
        model_state_dict.update(dict_revise)
        
        snn_q.load_state_dict(model_state_dict)

        ### Set threshold to 1 ###
        for name, module in snn_q.named_modules():
            if isinstance(module, (SpikeModule_q, SpikeModule_nq)):
                module.threshold = 1.
                
        ### Initialization of scale ###
        for p in snn_q.modules():
            if isinstance(p, (QConv2d, QLinear)):
                p.scale.data[0] = p.weight.abs().mean()
                                 
        snn_q.to(args.device)
        snn_q.set_spike_state(use_spike=True)
        
        ### Get parameters ###
        lamb, qweight, nqweight, otherparam, factor, b, scale, param_size = getparameters(snn_q)
                
        ### Optimizer ### 
        optimizer1 = None
        optimizer2 = None
        if args.opt == 'SGD':
            optimizer1 = torch.optim.SGD([{'params':qweight, 'lr':args.lr, 'weight_decay':args.wd},
                                          {'params':nqweight, 'lr':args.lr, 'weight_decay':args.wd},
                                          {'params':otherparam, 'lr':args.lr}], momentum=args.momentum)
            optimizer2 = torch.optim.Adam([{'params':lamb, 'lr':args.lr_lambda}])
        elif args.opt == 'Adam':
            optimizer1 = torch.optim.Adam([{'params':qweight, 'lr':args.lr},
                                          {'params':nqweight, 'lr':args.lr},
                                          {'params':otherparam, 'lr':args.lr}])
            optimizer2 = torch.optim.Adam([{'params':lamb, 'lr':args.lr_lambda}])
        else:
            raise NotImplementedError(args.opt)
            

        max_test_acc = 0
        
        ### Initialization of unconstrained window ###
        g = 1
        ucs = 1-1/g
        
        ### Initial update of multiplier ###
        updatelambda(optimizer2, qweight, lamb, scale, factor, b, ucs)
        
        ### Save epoch, train_acc, test_acc, train_loss, test_loss, cfs, lagsum_pre ###
        progress = np.zeros((1,7))
        
        ### lagsum_max, period ###
        lagsum_pre = 1e10 # lagsum_max in algorithm 1
        period = 0        
        print("Start Quantization")
        
        for epoch in range(args.epochs):
            start_time = time.time()
            lagsum = torch.zeros(1).to(args.device)
            snn_q.train()
            
            train_loss = 0
            train_acc = 0
            train_samples = 0
            
            with tqdm(total=len(train_loader)) as pbar:
                for idx, (images, labels) in enumerate(train_loader):
                    optimizer1.zero_grad()
                    images = images.to(args.device)
                    labels = labels.to(args.device)
                    
                    outputs = snn_q(images)
                    loss_network = F.cross_entropy(outputs, labels)
                    const = torch.zeros(1).to(args.device)
                    for i in range(len(lamb)):
                        const = const + constraints(qweight[i], lamb[i].detach(), scale[i], factor[i], b[i], ucs)
                    lag = loss_network + const
                    lagsum += lag.detach()

                    lag.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_value_(parameters=snn_q.parameters(), clip_value=1)
                    optimizer1.step()
                    for p in qweight:
                        p.data.clamp_(min=-1, max=1)

                    train_samples += labels.numel()
                    train_loss += loss_network.item() * labels.numel()
                    train_acc += (outputs.argmax(1) == labels).float().sum().item()
                    
                    if idx % 50 == 0:
                        print(f'samples/acc = {train_samples}/{train_acc}')

                    pbar.update(1)

            period += 1 

            train_loss /= train_samples
            train_acc /= train_samples

            print(f'epoch={epoch}...lagsum_pre={lagsum_pre}...lagsum={lagsum}')
            if lagsum >= lagsum_pre or period == args.period:
                print("Lambda update....")
                
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
                
            snn_q.eval()
            
            test_loss = 0
            test_acc = 0
            test_samples = 0
            
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(test_loader):
                    inputs = inputs.to(args.device)
                    targets = targets.to(args.device)
                    outputs = snn_q(inputs)
                    loss = F.cross_entropy(outputs, targets)
                    
                    test_samples += targets.numel()
                    test_loss += loss.item() * targets.numel()
                    test_acc += (outputs.argmax(1) == targets).float().sum().item()
                    
            test_loss /= test_samples
            test_acc /= test_samples    
            
            ### Calculate cfs ### 
            cfs = CFS(qweight,param_size, scale,factor, b)

            out_dir = './progress'
            
            ### Save data ###
            progress=np.append(progress,np.array([[epoch, train_acc, test_acc, train_loss, test_loss, cfs, lagsum_pre]]), axis=0)
            progress_data=pd.DataFrame(progress)
            progress_data.to_csv(out_dir+ f'/T_{args.T}_lr_{args.lr}_lrlambda_{args.lr_lambda}_wd_{args.wd}_period_{args.period}_{args.quant}_progress_{trial}.txt',
            index=False, header=False,sep='\t')
            
            save_max = False
            if test_acc > max_test_acc:
                max_test_acc = test_acc
                save_max = True
                print('save_max')
                
            checkpoint = {
                'net': snn_q.state_dict(),
                'factor' : factor,
                'scale'  : scale,
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
                torch.save(checkpoint, model_save_name_quant)               
                
            if epoch == 0 or (epoch+1) % 5 == 0:
                model_save_name_quant_interval = './logs/' + args.arch + '_' + args.dataset + '_' + 'T' + str(args.T) + '_' + args.opt + '_'\
                                           'lr' + str(args.lr) + 'lr_lambda' + str(args.lr_lambda) + 'wd' + str(args.wd) + '_' + \
                                            args.quant + '_' + str(trial) + '_' + f'{epoch}.pth'
                torch.save(checkpoint, model_save_name_quant_interval)
                
            total_time = time.time() - start_time
            print(f'epoch={epoch}, train_loss={train_loss}, train_acc={train_acc}, test_loss={test_loss}, test_acc={test_acc}, max_test_acc={max_test_acc}, total_time={total_time}, escape_time={(datetime.datetime.now()+datetime.timedelta(seconds=total_time * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}')
            
    