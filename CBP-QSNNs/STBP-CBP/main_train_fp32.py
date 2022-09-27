from tqdm import tqdm
import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from functions import *
from model import *

"""
python main_train_fp32.py --dataset CIFAR10 --mode train --decay 0.25 --thresh 0.5 --lens 0.5 --T 8
python main_train_fp32.py --dataset CIFAR100 --mode train --decay 0.8 --thresh 0.5 --lens 0.5 --T 8
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Pre-train')

parser.add_argument('--gpu', type=str, default="0", help='GPU id to use')
parser.add_argument('--dataset', type=str, default='CIFAR10', help='Which dataset to run (CIFAR10 or CIFAR100)')
parser.add_argument('--mode', type=str, default='train', help='Whether to train or eval')

parser.add_argument('--decay', type=float, default=0.25, help='Decay factor')
parser.add_argument('--thresh', type=float, default=0.5, help='Threshold voltage')
parser.add_argument('--lens', type=float, default=0.5, help='Hyper-parameter of surrogate gradient')
parser.add_argument('--T', type=int, default=8, help='Number of timesteps')

parser.add_argument('--num_workers', type=int, default=0, help='')
parser.add_argument('--batch_size', type=int, default=50, help='Batch size')
parser.add_argument('--num_epochs', type=int, default=100, help='Maximum number of epochs')
parser.add_argument('--opt', type=str, default='sgd', help='Optimizer (adam or sgd')
parser.add_argument('--learning_rate', type=float, default=3e-1, help='Initial learning rate')

args = parser.parse_args()
print(args)


def train(model, outs, train_dl, criterion, optimizer):
    loss_train = 0
    running_loss = 0
    
    with tqdm(total=len(train_dl)) as pbar:
        for i, (images, labels) in enumerate(train_dl):
            model.zero_grad()
            optimizer.zero_grad()

            images = images.to(device)
            outputs = model(images)
            labels_ = torch.zeros(args.batch_size, outs).scatter_(1, labels.view(-1, 1), 1)
                
            loss = criterion(outputs.cpu(), labels_)
            running_loss += loss.item() / len(train_dl)
            loss_train += loss.item() / len(train_dl)
            
            if (i+1) % 100 == 0:
                print(f'{int(i+1)} iterations ... running train loss:{running_loss}') 
                running_loss = 0
                
            loss.backward()
            optimizer.step()
            
            pbar.update(1)
            
    return loss_train


def test(model, outs, test_dl, criterion):
    loss_test = 0
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

            loss_test += loss.item() / len(test_dl)
            total += float(targets.size(0))
            correct += float(predicted.eq(targets).sum().item())
            
    acc = 100. * float(correct) / float(total)
    
    return loss_test, acc
    

def lr_scheduler(optimizer, epoch, lr_decay_epoch=40):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return optimizer



def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    out_dir = './checkpoint'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
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
        
    model = Conv_SNN(outs=outs, decay=args.decay, thresh=args.thresh, lens=args.lens, batch_size=args.batch_size, timesteps=args.T)
    model.to(device)
    
    criterion = nn.MSELoss()
    
    if args.mode == 'train':
        
        best_acc = 0
        best_epoch = 0
        acc_hist = list([])
        loss_train_hist = list([])
        loss_test_hist = list([])
        
        if args.opt == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        elif args.opt == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
        
        for epoch in range(args.num_epochs):
            loss_train = train(model, outs, train_dl, criterion, optimizer)
            optimizer = lr_scheduler(optimizer, epoch, 45)
            loss_test, acc = test(model, outs, test_dl, criterion)
            
            acc_hist.append(acc)
            loss_train_hist.append(loss_train)
            loss_test_hist.append(loss_test)
            
            print("Epoch: {}/{}.. ".format(epoch+1, args.num_epochs),
                  "Learning_rate: {}.. ".format(optimizer.param_groups[0]['lr']),
                  "Loss train: {:.5f}.. ".format(loss_train),
                  "Loss test: {:.5f}.. ".format(loss_test),
                  "Test Acc: {}..".format(acc))   
            
            state = {'epoch': epoch,
                     'net': model.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'loss_train_hist': loss_train_hist,
                     'loss_test_hist': loss_test_hist,
                     'test_acc_hist': acc_hist}
            
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                torch.save(state, out_dir + '/' + args.dataset + f'_best.pth')
            else:
                torch.save(state, out_dir + '/' + args.dataset + f'.pth')
    
    elif args.mode == 'eval':
        
        model_state_dict = torch.load('./trained_params/' + 'STBP_' + args.dataset + '_fp32_pretrained.pth') # from 'trained_params' directory
        model.load_state_dict(model_state_dict)
        loss_test, acc = test(model, outs, test_dl, criterion)
        print("Test Loss: {:.5f}.. ".format(loss_test), "Test Accuracy: {:.3f}".format(acc))
        

if __name__ == '__main__':
    main()
