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
## CIFAR10 (binary, ternary) ##
python main_quantize_cbp.py -config Networks/CIFAR10_bin.yaml -mode train
python main_quantize_cbp.py -config Networks/CIFAR10_ter.yaml -mode train
```


### CBP-QSNN-STBP
Working directory = CBP-QSNNs/CBP-QSNN-STBP/  
To train a CBP-QSNN-STBP on CIFAR10/100,  
- download fp32_pretrained models from [link](https://drive.google.com/drive/folders/1WOP2qFUCGkXJsMyzukqi2sDAZv4ol73g?usp=sharing) or get pretrained models by running main_train_fp32.py   
- save pretrained models to the trained_params directory.

```train
## CIFAR10 (binary, ternary) ##
python main_quantize_cbp.py --dataset CIFAR10 --mode train --decay 0.25 --thresh 0.5 --lens 0.5 --T 8 --quant bin
python main_quantize_cbp.py --dataset CIFAR10 --mode train --decay 0.25 --thresh 0.5 --lens 0.5 --T 8 --quant ter

## CIFAR100 (binary, ternary) ##
python main_quantize_cbp.py --dataset CIFAR100 --mode train --decay 0.8 --thresh 0.5 --lens 0.5 --T 8 --quant bin
python main_quantize_cbp.py --dataset CIFAR100 --mode train --decay 0.8 --thresh 0.5 --lens 0.5 --T 8 --quant ter
```

### CBP-QSNN-SNN-Calibration
Working directory = CBP-QSNNs/CBP-QSNN-SNN-Calibration/  
To train a CBP-QSNN-SNN-Calibration on CIFAR10/100,  
- download ann_fp32_pretrained models from [link](https://drive.google.com/drive/folders/19cAxdCJC8L531clVHAa9VlZqE3dqyVkt?usp=sharing) or get pretrained models from official implementation of [SNN-Calibration](https://github.com/yhhhli/SNN_Calibration)
- save pretrained models to the trained_params directory.

```train
## CIFAR10 (binary, ternary) ##
python main_calibration_quantize_cbp.py --dataset CIFAR10 --arch VGG16 --T 32 --calib light --dpath ./datasets --device cuda:0 --opt SGD --quant bin --lr 1e-2 --lr_lambda 1e-3 
python main_calibration_quantize_cbp.py --dataset CIFAR10 --arch VGG16 --T 32 --calib light --dpath ./datasets --device cuda:0 --opt SGD --quant ter --lr 1e-2 --lr_lambda 1e-3

## CIFAR100 (binary, ternary) ##
python main_calibration_quantize_cbp.py --dataset CIFAR100 --arch VGG16 --T 32 --calib light --dpath ./datasets --device cuda:0 --opt SGD --quant bin --lr 1e-2 --lr_lambda 1e-3 
python main_calibration_quantize_cbp.py --dataset CIFAR100 --arch VGG16 --T 32 --calib light --dpath ./datasets --device cuda:0 --opt SGD --quant ter --lr 1e-2 --lr_lambda 1e-3
```



## Evaluation

To evaluate a DTS-SNN with zero sum temporal kernel on DVS128-Gesture or N-Cars or SHD, run this command:
```evaluation
python main.py --dataset DVS128-Gesture --temporal_kernel ktzs --ds 1 --dt 5 --T 300 --batch_size 16 --gpu 0 1 --mode eval
python main.py --dataset N-Cars --temporal_kernel ktzs --dt 1 --T 100 --num_workers 0 --batch_size 64 --gpu 0 1 --mode eval
python main.py --dataset SHD --temporal_kernel ktzs --dt 1 --T 500 --num_workers 0 --batch_size 256 --gpu 0 1 --mode eval
```

To evaluate a DTS-SNN with single exponential temporal kernel on DVS128-Gesture or N-Cars or SHD, run this command:
```evaluation
python main.py --dataset DVS128-Gesture --temporal_kernel kt --ds 1 --dt 5 --T 300 --batch_size 16 --gpu 0 1 --mode eval
python main.py --dataset N-Cars --temporal_kernel kt --dt 1 --T 100 --num_workers 0 --batch_size 64 --gpu 0 1 --mode eval
python main.py --dataset SHD --temporal_kernel kt --dt 1 --T 500 --num_workers 0 --batch_size 256 --gpu 0 1 --mode eval
```

## Results
Our model achieves the following performance on: 

- DVS128-Gesture dataset

| Method                                            | Network                    | Accuracy (%) |
| ------------------------------------------------- | -------------------------- | ------------ |
| DTS-SNN (with zero sum temporal kernel)           | 3136-400-11 (FCN)          | 96.06%       |
| DTS-SNN (with single exponetial temporal kernel)  | 3136-400-11 (FCN)          | 92.94%       |

- N-Cars dataset

| Method                                            | Network                    | Accuracy (%) |
| ------------------------------------------------- | -------------------------- | ------------ |
| DTS-SNN (with zero sum temporal kernel)           | 3000-400-2 (FCN)           | 90.28%       |
| DTS-SNN (with single exponetial temporal kernel)  | 3000-400-2 (FCN)           | 89.47%       |

- SHD dataset

| Method                                            | Network                    | Accuracy (%) |
| ------------------------------------------------- | -------------------------- | ------------ |
| DTS-SNN (with zero sum temporal kernel)           | 105-128-2 (FCN)            | 82.17%       |
| DTS-SNN (with single exponetial temporal kernel)  | 105-128-2 (FCN)            | 66.21%       |
