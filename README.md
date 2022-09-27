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

2. To train a STBP+CBP on CIFAR10/100, run this command:
```train
## CIFAR10 ##
python main_quantize_cbp.py --dataset CIFAR10 --mode train --decay 0.25 --thresh 0.5 --lens 0.5 --T 8 --quant bin
python main_quantize_cbp.py --dataset CIFAR10 --mode train --decay 0.25 --thresh 0.5 --lens 0.5 --T 8 --quant ter

## CIFAR100 ##
python main_quantize_cbp.py --dataset CIFAR100 --mode train --decay 0.8 --thresh 0.5 --lens 0.5 --T 8 --quant bin
python main_quantize_cbp.py --dataset CIFAR100 --mode train --decay 0.8 --thresh 0.5 --lens 0.5 --T 8 --quant ter
```

To train a DTS-SNN with single exponential temporal kernel on DVS128-Gesture or N-Cars or SHD, run this command:
```train
python main.py --dataset DVS128-Gesture --temporal_kernel kt --ds 1 --dt 5 --T 300 --batch_size 16 --gpu 0 1 --mode train
python main.py --dataset N-Cars --temporal_kernel kt --dt 1 --T 100 --num_workers 0 --batch_size 64 --gpu 0 1 --mode train
python main.py --dataset SHD --temporal_kernel kt --dt 1 --T 500 --num_workers 0 --batch_size 256 --gpu 0 1 --mode train
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
