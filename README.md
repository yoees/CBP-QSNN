# CBP-QSNN
This repository is the official implementation of **CBP-QSNN: Spiking Neural Networks quantized using constrained backpropagation**.

## Environment
Create virtual environment:
```setup
conda create -n name python=3.8.12
conda activate name
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
```
To install requirements:
```setup
pip install -r requirements.txt
```



## Training

### CBP-QSNN-TSSL-BP
Working directory = CBP-QSNNs/CBP-QSNN-TSSL-BP/  
To train a CBP-QSNN-TSSL-BP on CIFAR10,  
- download fp32_pretrained models from [link](https://drive.google.com/drive/folders/10XZeHHdLH82All1QJAcorJyWHtoUyrCv?usp=sharing) or get pretrained models from official implementation of [TSSL-BP](https://github.com/stonezwr/TSSL-BP)
- save pretrained models to the trained_params directory.

```train
## AlexNet on CIFAR10 (binary, ternary) ##
python main_quantize_cbp.py -config Networks/CIFAR10_bin.yaml -mode train
python main_quantize_cbp.py -config Networks/CIFAR10_ter.yaml -mode train
```

### CBP-QSNN-STBP
Working directory = CBP-QSNNs/CBP-QSNN-STBP/  
To train a CBP-QSNN-STBP on CIFAR10/100,  
- download fp32_pretrained models from [link](https://drive.google.com/drive/folders/1WOP2qFUCGkXJsMyzukqi2sDAZv4ol73g?usp=sharing) or get pretrained models by running main_train_fp32.py   
- save pretrained models to the trained_params directory.

```train
## 7Conv,3FC on CIFAR10 (binary, ternary) ##
python main_quantize_cbp.py --dataset CIFAR10 --mode train --decay 0.25 --thresh 0.5 --lens 0.5 --T 8 --quant bin
python main_quantize_cbp.py --dataset CIFAR10 --mode train --decay 0.25 --thresh 0.5 --lens 0.5 --T 8 --quant ter

## 7Conv,3FC on CIFAR100 (binary, ternary) ##
python main_quantize_cbp.py --dataset CIFAR100 --mode train --decay 0.8 --thresh 0.5 --lens 0.5 --T 8 --quant bin
python main_quantize_cbp.py --dataset CIFAR100 --mode train --decay 0.8 --thresh 0.5 --lens 0.5 --T 8 --quant ter
```

### CBP-QSNN-SEW-ResNet
Datasets = DVS128Gesture, CIFAR10DVS, ImageNet  
Working directory = CBP-QSNNs/CBP-QSNN-SEW-ResNet/dataset_name  
To train a CBP-QSNN-SEW-ResNet on DVS128Gesture/CIFAR10DVS,  
- download fp32_pretrained models from [link](https://drive.google.com/drive/folders/1nq5NMVrlxlsjM2yd3GpYW3O5MLuHSC7j?usp=sharing) or get pretrained models from official implementation of [SEW-ResNet](https://github.com/fangwei123456/Spike-Element-Wise-ResNet)   
- save pretrained models to the trained_params directory.

```train
## 7B-Net on DVS128Gesture (binary, ternary) ##
python main_quantize_cbp.py --tb --amp --output-dir ./logs --model SEWResNet --connect_f ADD --device cuda:0 --epoch 200 --T_train 12 --T 16 --data-path ./datasets/DVS128Gesture --lr 0.1 --lr-lambda 0.01 --quant bin --period 20
python main_quantize_cbp.py --tb --amp --output-dir ./logs --model SEWResNet --connect_f ADD --device cuda:0 --epoch 200 --T_train 12 --T 16 --data-path ./datasets/DVS128Gesture --lr 0.1 --lr-lambda 0.01 --quant ter --period 20

## Wide-7B-Net on CIFAR10DVS (binary, ternary) ##
python main_quantize_cbp.py -amp -out_dir ./logs -model SEWResNet -cnf ADD -device cuda:0 -opt SGD -lr 0.1 -lr_lambda 0.01 -epochs 64 -quant bin -period 20
python main_quantize_cbp.py -amp -out_dir ./logs -model SEWResNet -cnf ADD -device cuda:0 -opt SGD -lr 0.1 -lr_lambda 0.01 -epochs 64 -quant ter -period 20
```

To train a CBP-QSNN-SEW-ResNet on ImageNet,
- get pretrained models('sew18_checkpoint_319.pth' and 'sew34_checkpoint_319.pth') from official implementation of [SEW-ResNet](https://github.com/fangwei123456/Spike-Element-Wise-ResNet)   
- save pretrained models to the trained_params directory.

```train
## SEW-ResNet18 on ImageNet (binary, ternary) ##
python -m torch.distributed.launch --nproc_per_node=2 --use_env main_quantize_cbp.py --model sew_resnet18 -b 32 --output-dir ./logs --tb --print-freq 128 --amp --cache-dataset --connect_f ADD --T 4 --lr 0.1 --lr_lambda 0.01 --epochs 100 --data-path ./datasets/imagenet --quant bin --period 20 --device cuda
python -m torch.distributed.launch --nproc_per_node=2 --use_env main_quantize_cbp.py --model sew_resnet18 -b 32 --output-dir ./logs --tb --print-freq 128 --amp --cache-dataset --connect_f ADD --T 4 --lr 0.1 --lr_lambda 0.01 --epochs 100 --data-path ./datasets/imagenet --quant ter --period 20 --device cuda

## SEW-ResNet34 on ImageNet (binary, ternary) ##
python -m torch.distributed.launch --nproc_per_node=2 --use_env main_quantize_cbp.py --model sew_resnet34 -b 32 --output-dir ./logs --tb --print-freq 128 --amp --cache-dataset --connect_f ADD --T 4 --lr 0.1 --lr_lambda 0.01 --epochs 100 --data-path ./datasets/imagenet --quant bin --period 20 --device cuda
python -m torch.distributed.launch --nproc_per_node=2 --use_env main_quantize_cbp.py --model sew_resnet34 -b 32 --output-dir ./logs --tb --print-freq 128 --amp --cache-dataset --connect_f ADD --T 4 --lr 0.1 --lr_lambda 0.01 --epochs 100 --data-path ./datasets/imagenet --quant ter --period 20 --device cuda
```

### CBP-QSNN-SNN-Calibration
Working directory = CBP-QSNNs/CBP-QSNN-SNN-Calibration/  
To train a CBP-QSNN-SNN-Calibration on CIFAR10/100,  
- download ann_fp32_pretrained models from [link](https://drive.google.com/drive/folders/19cAxdCJC8L531clVHAa9VlZqE3dqyVkt?usp=sharing) or get pretrained models from official implementation of [SNN-Calibration](https://github.com/yhhhli/SNN_Calibration)
- save pretrained ann models to the trained_params directory.

```train
## VGG16 on CIFAR10 (binary, ternary) ##
python main_calibration_quantize_cbp.py --dataset CIFAR10 --arch VGG16 --T 32 --calib light --dpath ./datasets --device cuda:0 --opt SGD --quant bin --lr 1e-2 --lr_lambda 1e-3 
python main_calibration_quantize_cbp.py --dataset CIFAR10 --arch VGG16 --T 32 --calib light --dpath ./datasets --device cuda:0 --opt SGD --quant ter --lr 1e-2 --lr_lambda 1e-3

## VGG16 on CIFAR100 (binary, ternary) ##
python main_calibration_quantize_cbp.py --dataset CIFAR100 --arch VGG16 --T 32 --calib light --dpath ./datasets --device cuda:0 --opt SGD --quant bin --lr 1e-2 --lr_lambda 1e-3 
python main_calibration_quantize_cbp.py --dataset CIFAR100 --arch VGG16 --T 32 --calib light --dpath ./datasets --device cuda:0 --opt SGD --quant ter --lr 1e-2 --lr_lambda 1e-3
```
