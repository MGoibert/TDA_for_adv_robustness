import os
import sys
import numpy as np
import mlflow

mlflow.set_tracking_uri("https://mlflow.par.prod.crto.in")
mlflow.set_experiment("tda_adv_detection")

path = "/home/t.ricatte/dev/tda/tda/experiments/lid/lid_binary.py"
dataset = " --dataset CIFAR100"
archi = " --architecture cifar100_resnet56"
epochs = " --epochs 42"
attack_backend = " --attack_backend FOOLBOX"
dataset_size = " --dataset_size 500"
all_eps = ' --all_epsilons "0.1"'
attack_type = " --attack_type PGD"
perc_of_nn = " --perc_of_nn 0.3"
batch_size = " --batch_size 100"

command = (
    sys.executable
    + " "
    + path
    + dataset
    + archi
    + epochs
    + attack_backend
    + dataset_size
    + all_eps
    + attack_type
    + perc_of_nn
    + batch_size
)
os.system(command)
