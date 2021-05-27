import os
import sys
import numpy as np


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
all_eps = ' --all_epsilons "1.0"'
njobs = " --n_jobs 4"
attack_type = " --attack_type CW"

list_layers = [[21, 17, 19, 24, 26, 29, 31], [38, 34, 36, 41, 43, 46, 48]]
myts = np.linspace(0.001, 0.4, 10)
thresholds_list = list()

for layers in list_layers:
    for myt in myts:
        mystr_ = " --thresholds "
        the_thresh = get_threshold(layers, myt)
        thresholds_list.append(mystr_ + the_thresh)

for thresholds in thresholds_list:
    command = (
        sys.executable
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
