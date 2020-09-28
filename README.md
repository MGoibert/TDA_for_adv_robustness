This repository contains the code associated with the submission.

## 1) Getting started

### A) Setup

*  Create a venv with python 3.6 and activate it
```bash
$ python -m venv my_env
$ source ./my_env/bin/activate
```
* Install the project in the env (you should be in the root folder of the project)
```bash
$ pip install -e .
``` 
 
 Note that we use dionysus to compute the persistent diagrams. This library requires boost (see the official doc)
 
## 2) Launching experiments
 
### A) Detections

#### 0. Common 

All our scripts have some common arguments:
* attack_type (can be typically FGSM, PGD, CW, BOUNDARY)
* dataset (can be MNIST, SVHN, CIFAR10)
* architecture (see below)
* epochs (number of epochs for the architecture ; see below)

#### 1. LID Baseline

The specific arguments for LID are the batch_size and the percentage of nearest neighbors for the LID estimation.

Example command:

```bash
python tda/experiments/lid/lid_binary.py \
    --attack_type PGD \
    --architecture cifar_resnet_1 \
    --dataset CIFAR10 \
    --epochs 100 \
    --perc_of_nn 0.3 \
    --batch_size 100 
```
