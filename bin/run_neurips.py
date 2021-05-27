import os
import sys
import numpy as np
import mlflow

mlflow.set_tracking_uri("https://mlflow.par.prod.crto.in")
mlflow.set_experiment("tda_adv_detection")

def get_threshold(list_edges, myt=0.01):
    rep = list()
    for t in list_edges:
        r = str(t) + ":0:" + str(myt)
        rep.append(r)
    reps = "_".join(rep)
    return reps


path = "/home/t.ricatte/dev/tda/tda/experiments/ours/our_binary.py"
dataset = " --dataset CIFAR100"
archi = " --architecture cifar100_resnet56"
epochs = " --epochs 42"
threshold_strat = " --threshold_strategy UnderoptimizedLargeFinal"
attack_backend = " --attack_backend FOOLBOX"
dataset_size = " --dataset_size 500"
all_eps = ' --all_epsilons "0.01;0.05;0.1"'
njobs = " --n_jobs 1"
attack_type = " --attack_type PGD"

list_layers = [
    [51, 47, 49, 54, 56, 59, 61],
    [64, 66, 69, 71, 74, 76],
    [79, 81, 84, 86, 89, 91],
    [98, 94, 96, 101, 103, 106, 108],
    [111, 113, 116, 118, 121, 123],
    [126, 128, 131, 133, 136, 138]
]
myts = np.linspace(0.1, 0.4, 3)
thresholds_list = list()

for layers in list_layers:
    for myt in myts:
        mystr_ = " --thresholds "
        the_thresh = get_threshold(layers, myt)
        thresholds_list.append(mystr_ + the_thresh)


for thresholds in thresholds_list:
    command = (
        sys.executable
        + " "
        + path
        + dataset
        + archi
        + epochs
        + threshold_strat
        + attack_backend
        + dataset_size
        + all_eps
        + attack_type
        + njobs
        + thresholds
    )
    os.system(command)
