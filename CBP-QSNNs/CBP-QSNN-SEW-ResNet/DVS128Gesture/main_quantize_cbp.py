import datetime
import os
import time
import pandas as pd
import torch
import torch.utils.data
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import math
from torch.cuda import amp
import q_smodels, utils
from spikingjelly.clock_driven import functional
from spikingjelly.datasets import dvs128_gesture
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


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, print_freq, scaler, T_train,
                   qweight, lamb, scale, factor, b, ucs, param_size):
    
    lagsum = torch.zeros(1).to(device)
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', utils.SmoothedValue(window_size=10, fmt='{value}'))

    header = 'Epoch: [{}]'.format(epoch)

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        image = image.float()  # [N, T, C, H, W]

        if T_train:
            sec_list = np.random.choice(image.shape[1], T_train, replace=False)
            sec_list.sort()
            image = image[:, sec_list]

        if scaler is not None:
            with amp.autocast():
                output = model(image)
                loss_network = criterion(output, target)

                const = torch.zeros(1).to(device)
                for i in range(len(lamb)):
                    const = const + constraints(qweight[i], lamb[i].detach(), scale[i], factor[i], b[i], ucs)
                lag = loss_network + const
                lagsum += lag.detach()
        else:
            output = model(image)
            loss_network = criterion(output, target)

            const = torch.zeros(1).to(device)
            for i in range(len(lamb)):
                const = const + constraints(qweight[i], lamb[i].detach(), scale[i], factor[i], b[i], ucs)
            lag = loss_network + const
            lagsum += lag.detach()

        optimizer.zero_grad()

        if scaler is not None:

            scaler.scale(lag).backward(retain_graph=True)
            torch.nn.utils.clip_grad_value_(parameters=model.parameters(), clip_value=1)
            scaler.step(optimizer)
            scaler.update()
            for p in qweight:
                p.data.clamp_(min=-1, max=1)

        else:

            lag.backward(retain_graph=True)
            torch.nn.utils.clip_grad_value_(parameters=model.parameters(), clip_value=1)
            optimizer.step()
            for p in qweight:
                p.data.clamp_(min=-1, max=1)

        functional.reset_net(model)

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        loss_s = loss_network.item()
        if math.isnan(loss_s):
            raise ValueError('loss is Nan')
        acc1_s = acc1.item()
        acc5_s = acc5.item()

        metric_logger.update(loss=loss_s, lr=optimizer.param_groups[0]["lr"])

        metric_logger.meters['acc1'].update(acc1_s, n=batch_size)
        metric_logger.meters['acc5'].update(acc5_s, n=batch_size)
        metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    return lagsum, metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg


def evaluate(model, criterion, data_loader, device, print_freq=100, header='Test:'):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            image = image.float()
            output = model(image)
            loss = criterion(output, target)
            functional.reset_net(model)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    loss, acc1, acc5 = metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg
    print(f' * Acc@1 = {acc1}, Acc@5 = {acc5}, loss = {loss}')
    return loss, acc1, acc5

def load_data(dataset_dir, distributed, T):
    # Data loading code
    print("Loading data")

    st = time.time()

    dataset_train = dvs128_gesture.DVS128Gesture(root=dataset_dir, train=True, data_type='frame', frames_number=T, split_by='number')
    dataset_test = dvs128_gesture.DVS128Gesture(root=dataset_dir, train=False, data_type='frame', frames_number=T,
                                                 split_by='number')

    print("Took", time.time() - st)

    print("Creating data loaders")
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset_train)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset_train, dataset_test, train_sampler, test_sampler

def main(args):

    max_test_acc1 = 0.
    test_acc5_at_max_test_acc1 = 0.

    train_tb_writer = None
    te_tb_writer = None

    utils.init_distributed_mode(args)
    print(args)
    
    if args.test_only:
        pass
    else:
        output_dir = os.path.join(args.output_dir, f'{args.model}_b{args.batch_size}_T{args.T}')  # default directory is './logs'

        if args.T_train:
            output_dir += f'_Ttrain{args.T_train}'

        if args.weight_decay:
            output_dir += f'_wd{args.weight_decay}'

        if args.epochs:
            output_dir += f'_epoch{args.epochs}'

        if args.adam:
            output_dir += '_adam'
        else:
            output_dir += '_sgd'

        if args.connect_f:
            output_dir += f'_cnf_{args.connect_f}'

        if args.period:
            output_dir += f'_period{args.period}'

        if args.quant:
            output_dir += f'_{args.quant}'

        if not os.path.exists(output_dir):
            utils.mkdir(output_dir)

        output_dir = os.path.join(output_dir, f'lr{args.lr}_lrlamba{args.lr_lambda}')
        if not os.path.exists(output_dir):
            utils.mkdir(output_dir)
    
    torch.cuda.set_device(args.device)
    device = torch.device(args.device)
    
    data_path = args.data_path

    dataset_train, dataset_test, train_sampler, test_sampler = load_data(data_path, args.distributed, args.T)
    print(f'dataset_train:{dataset_train.__len__()}, dataset_test:{dataset_test.__len__()}')

    data_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers, pin_memory=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size,
        sampler=test_sampler, num_workers=args.workers, pin_memory=True)
    
    ### Define model ###
    model = q_smodels.__dict__[args.model](args.connect_f, args.quant)
    print("Creating model")
    model.to(device)
    
    ### Get pre-trained weight ###
    checkpoint_path = './trained_params/SEW_ResNet_DVS128Gesture_fp32_pretrained.pth' # from 'trained_params' directory
    checkpoint = torch.load(checkpoint_path)
    model_state_dict = model.state_dict()
    model_state_dict.update(checkpoint)
    model.load_state_dict(model_state_dict)
    print("Loading model")
    
    ### Initialization of scale ###
    for p in model.modules():
        if isinstance(p,(QConv2d, QLinear)):
            p.scale.data[0] = p.weight.abs().mean()
            
    ### Get parameters ###
    lamb, qweight, nqweight, otherparam, timeconstant, factor, b, scale, param_size = getparameters(model) 

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = nn.CrossEntropyLoss()
     
    ### Optimizer ###    
    if args.adam:
        optimizer1 = torch.optim.Adam([{'params':qweight, 'lr':args.lr},
                                       {'params':nqweight, 'lr':args.lr},
                                       {'params':otherparam, 'lr':args.lr},
                                       {'params':timeconstant, 'lr':args.lr}])
        optimizer2 = torch.optim.Adam([{'params':lamb, 'lr':args.lr_lambda}])
    else:
        optimizer1 = torch.optim.SGD([{'params':qweight, 'lr':args.lr},
                                      {'params':nqweight, 'lr':args.lr},
                                      {'params':otherparam, 'lr':args.lr},
                                      {'params':timeconstant, 'lr':args.lr}], momentum=0.9)
        optimizer2 = torch.optim.Adam([{'params':lamb, 'lr':args.lr_lambda}])

    if args.amp:
        scaler = amp.GradScaler()
    else:
        scaler = None

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer1.load_state_dict(checkpoint['optimizer1'])
        optimizer2.load_state_dict(checkpoint['optimizer2'])
        args.start_epoch = checkpoint['epoch'] + 1
        lamb = checkpoint['lamb']
        lagsum_pre = checkpoint['lagsum_pre']
        period = checkpoint['period']
        ucs = checkpoint['ucs']
        g = checkpoint['g']
        progress = checkpoint['progress']
        max_test_acc1 = checkpoint['max_test_acc1']
        test_acc5_at_max_test_acc1 = checkpoint['test_acc5_at_max_test_acc1']

    if args.test_only:
        model_state_dict = torch.load('./trained_params/SEW_ResNet_DVS128Gesture_' + args.quant + '_cbp_prequantized.pth')  # from 'trained_params' directory
        model.load_state_dict(model_state_dict)
        evaluate(model, criterion, data_loader_test, device=device, header='Test:')

        return

    if args.tb and utils.is_main_process():
        purge_step_train = args.start_epoch
        purge_step_te = args.start_epoch
        train_tb_writer = SummaryWriter(output_dir + '_logs/train', purge_step=purge_step_train)
        te_tb_writer = SummaryWriter(output_dir + '_logs/te', purge_step=purge_step_te)
        with open(output_dir + '_logs/args.txt', 'w', encoding='utf-8') as args_txt:
            args_txt.write(str(args))

        print(f'purge_step_train={purge_step_train}, purge_step_te={purge_step_te}')
    
    if args.resume:       
        pass
    else:       
        ### Initialization of unconstrained window ###
        g = 1
        ucs = 1-1/g

        ### Initial update of multiplier ###
        updatelambda(optimizer2, qweight, lamb, scale, factor, b, ucs)

        ### Save epoch, top1, top5 accuracy, cfs and lagsum_pre ###
        progress=np.zeros((1,5))

        ### lagsum_max, period ###
        lagsum_pre = 1e10  # lagsum_max in algorithm 1 
        period = 0
    
    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        save_max = False
        if args.distributed:
            train_sampler.set_epoch(epoch)
        lagsum, train_loss, train_acc1, train_acc5 = train_one_epoch(model, criterion, optimizer1, data_loader, device, epoch, args.print_freq, scaler, args.T_train,
                                                                     qweight, lamb, scale, factor, b, ucs, param_size)
        period += 1
        
        if utils.is_main_process():
            train_tb_writer.add_scalar('lagsum', lagsum, epoch)
            train_tb_writer.add_scalar('train_loss', train_loss, epoch)
            train_tb_writer.add_scalar('train_acc1', train_acc1, epoch)
            train_tb_writer.add_scalar('train_acc5', train_acc5, epoch)

        if  lagsum >= lagsum_pre or period == args.period:
            print('lambda update')
            
            ### Update of unconstrained window ###
            if g<10:
                g+=1
            else:
                g+=10
            ucs =1-1/g

            ### Learning rate scheduler ###
            if g==20:
                adjust_lr(optimizer1, 0.1)

            ### Update of lambda ###
            updatelambda(optimizer2, qweight, lamb, scale, factor, b, ucs)
            
            ### Reset lagsum and period ###
            lagsum_pre = 1E10
            period = 0

        else:
            lagsum_pre = lagsum.item()
        
        test_loss, test_acc1, test_acc5 = evaluate(model, criterion, data_loader_test, device=device, header='Test:')
        if te_tb_writer is not None:
            if utils.is_main_process():

                te_tb_writer.add_scalar('test_loss', test_loss, epoch)
                te_tb_writer.add_scalar('test_acc1', test_acc1, epoch)
                te_tb_writer.add_scalar('test_acc5', test_acc5, epoch)

        if max_test_acc1 < test_acc1:
            max_test_acc1 = test_acc1
            test_acc5_at_max_test_acc1 = test_acc5
            save_max = True
             
        ### Calculate cfs ###  
        cfs=CFS(qweight,param_size, scale,factor, b)
        
        ### Save data ###
        progress=np.append(progress,np.array([[epoch, test_acc1, test_acc5, cfs, lagsum_pre]]),axis=0)
        progress_data=pd.DataFrame(progress)
        progress_data.to_csv(output_dir+"/progress.txt",
        index=False, header=False,sep='\t')

        if output_dir:

            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer1': optimizer1.state_dict(),
                'optimizer2': optimizer2.state_dict(),
                'epoch': epoch,
                'lamb': lamb,
                'lagsum_pre':lagsum_pre,
                'period':period,
                'ucs':ucs,
                'g':g,
                'progress':progress,
                'args': args,
                'max_test_acc1': max_test_acc1,
                'test_acc5_at_max_test_acc1': test_acc5_at_max_test_acc1,
            }
            
            if epoch ==0 or (epoch+1) % 40 == 0:
                utils.save_on_master(
                    checkpoint,
                    os.path.join(output_dir, f'checkpoint_{epoch}.pth'))
                
            if save_max:
                utils.save_on_master(
                    checkpoint,
                    os.path.join(output_dir, 'checkpoint_max_test_acc1.pth'))
                
        print(args)
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))

        print('Training time {}'.format(total_time_str), 'max_test_acc1', max_test_acc1, 'test_acc5_at_max_test_acc1', test_acc5_at_max_test_acc1)
        print(output_dir)
        
    if output_dir:
        utils.save_on_master(
            checkpoint,
            os.path.join(output_dir, f'checkpoint_{epoch}.pth'))

    return max_test_acc1



def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')
    parser.add_argument('--model', help='model')

    parser.add_argument('--data-path', help='dataset')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=16, type=int)
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--lr', type=float, help='initial learning rate of weight update', default=0.1)
    parser.add_argument('--lr-lambda', type=float, help='initial learning rate of lambda update', default=0.01)
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='Momentum for SGD. Adam will not use momentum')
    parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                        metavar='W', help='weight decay (default: 0)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=64, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='./logs', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )

    # Mixed precision training parameters
    parser.add_argument('--amp', action='store_true',
                        help='Use AMP training')


    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--tb', action='store_true',
                        help='Use TensorBoard to record logs')
    parser.add_argument('--T', default=16, type=int, help='simulation steps')
    parser.add_argument('--adam', action='store_true',
                        help='Use Adam')

    parser.add_argument('--connect_f', type=str, help='element-wise connect function')
    parser.add_argument('--quant', default='bin', type=str, help='weight quantization (bin or ter)')
    parser.add_argument('--period', default=20, type=int, help='max period of lambda update')
    parser.add_argument('--T_train', type=int)

    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parse_args()
    main(args)
