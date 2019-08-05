#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 12:30:59 2019

Author: Morgane Goibert <morgane.goibert@gmail.com>
"""

from IPython.display import Image
from os import chdir
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import dionysus as d
import sys
from scipy.spatial.distance import squareform

import scipy.stats as stats
import scipy.sparse as sparse
from numpy import inf
from scipy.linalg import block_diag
import pandas as pd
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
from operator import itemgetter
from tqdm import tqdm
import time
import logging
mpl_logger = logging.getLogger('matplotlib') 
mpl_logger.setLevel(logging.WARNING) 
torch.set_default_tensor_type(torch.DoubleTensor)
sns.set()

# Files from the project to import
from NN_architecture import MNISTMLP
from datasets import (train_loader_MNIST, test_loader_MNIST, val_loader_MNIST,
                      test_set)
from functions import (train_NN, compute_val_acc, compute_test_acc,
                       compute_persistent_dgm, get_class_indices,
                       compute_intra_distances, compute_distances,
                       produce_dgms, save_result)
from utils import parse_cmdline_args


# ------------------------
# ----- Command line arguments
# ------------------------

args = parse_cmdline_args()
num_epochs = args.num_epochs
epsilon = args.epsilon
noise = args.noise
threshold = args.threshold
n = args.num_computation
save = args.save

# --------------------
# ----- Neural Network
# --------------------

# Use the MLP model
model = MNISTMLP()

# MNIST dataset
train_loader = train_loader_MNIST
test_loader = test_loader_MNIST
val_loader = val_loader_MNIST
test_set = test_set

# Train the NN
num_epochs = 15
loss_func = nn.CrossEntropyLoss()
net = train_NN(model, train_loader, val_loader, loss_func, num_epochs)[0]

# Compute accuarcies
compute_val_acc(model, val_loader)
compute_test_acc(model, test_loader)

# Test how it is working
#dgms1 = compute_persistent_dgm(net, test_set, loss_func, numero_ex=0,
#                               adversarial=False, threshold=12000)
#d.plot.plot_diagram(dgms1[0][0], show = True)


def run_dist_detection(n, threshold, epsilon, noise, start_n=0, net=net, test_set=test_set,
                       loss_func=loss_func, num_classes=None):
    all_class = range(10)
    
    # Indices for all classes
    ind_classes = {key: get_class_indices(key, number="all") for key in all_class}
    
    # ---------------------------
    ### Step 1: Clean inputs part
    # ---------------------------
    
    # Indices for clean computation
    inds_all_class = {key: ind_classes[key][start_n:n+start_n] for key in all_class}
    
    # Dict containing the persistend dgm for clean inputs (and other info),
    # organized by observed class (i.e. in each class, we have the persistent dgm
    # for the inputs classified correclty or not in this class)
    dgms_dict = produce_dgms(net, test_set, loss_func, threshold, inds_all_class,
                             adversarial=False, epsilon=epsilon, noise=0,
                             num_classes=num_classes)

    # Intra-class distance distribution
    distrib_dist = compute_intra_distances(dgms_dict)
    
    # -------------------------
    ### Step 2: Adv inputs part
    # -------------------------
    
    # n indices for every class
    inds_all_class_adv = {key: ind_classes[key][n+start_n:2*n+start_n] for key in all_class}
    
    # Dict containing the persistend dgm for adv. inputs (and other info),
    # organized by observed class (i.e. in each class, we have the persistent dgm
    # for the inputs classified correclty or not in this class)
    dgms_dict_adv = produce_dgms(net, test_set, loss_func, threshold, inds_all_class_adv,
                                 adversarial=True, epsilon=epsilon, noise=0,
                                 num_classes=num_classes)
    
    # dictionary containing only incorrectly predicted adv. inputs
    dgms_dict_adv_incorrect = copy.deepcopy(dgms_dict_adv)
    for key in list(dgms_dict_adv_incorrect.keys()):
        for ind in list(dgms_dict_adv_incorrect[key].keys()):
            if dgms_dict_adv_incorrect[key][ind][1] == dgms_dict_adv_incorrect[key][ind][3]:
                del dgms_dict_adv_incorrect[key][ind]

    # Distance distribution for adv. inputs vs clean inputs
    distrib_dist_adv = compute_distances(dgms_dict, dgms_dict_adv)
    distrib_dist_adv_incorrect = compute_distances(dgms_dict, dgms_dict_adv_incorrect)
    
    # ---------------------------
    ### Step 3: Noisy inputs part
    # ---------------------------
    
    # n indices for every class
    inds_all_class_noise = {key: ind_classes[key][2*n+start_n:3*n+start_n] for key in all_class}

    # Dict containing the persistend dgm for noisy inputs (and other info),
    # organized by observed class (i.e. in each class, we have the persistent dgm
    # for the inputs classified correclty or not in this class)
    dgms_dict_noise = produce_dgms(net, test_set, loss_func, threshold, inds_all_class_noise,
                                   adversarial=False, epsilon=epsilon, noise=noise,
                                   num_classes=num_classes)
      
    # Distance distribution for adv. inputs vs clean inputs
    distrib_dist_noise = compute_distances(dgms_dict, dgms_dict_noise,
                                           adversarial=False, noisy=True)
    
    # ---------------
    ### Step 4: Plots
    # ---------------
    
    for i in all_class:
        sns.distplot(distrib_dist["dist_"+str(i)+"_"+str(i)], hist=False, label="clean "+str(i))
        sns.distplot(distrib_dist_adv["dist_adv_"+str(i)], hist=False, label="adv "+str(i))
        sns.distplot(distrib_dist_adv_incorrect["dist_adv_"+str(i)], hist=False, label="adv incorrect "+str(i))
        sns.distplot(distrib_dist_noise["dist_noise_"+str(i)], hist=False, label="noise "+str(i))
        plt.show()
    
    return (inds_all_class, dgms_dict, distrib_dist,
            inds_all_class_adv, dgms_dict_adv, distrib_dist_adv,
            dgms_dict_adv_incorrect, distrib_dist_adv_incorrect,
            inds_all_class_noise, dgms_dict_noise, distrib_dist_noise)

# ----------------
# ----------------
# Run experiment !
# ----------------
# ----------------

n = 50
threshold = 20000
epsilon = 0.075
noise=0.075
start_n = 0
num_classes=10

# Experiment with all (clean, adv, noisy) inputs
result = run_dist_detection(n, threshold, epsilon, noise, start_n=start_n,
                            net=net, test_set=test_set, loss_func=loss_func,
                            num_classes=num_classes)

# Save
# save_result(result, threshold, epsilon, noise)

distrib_dist = result[2]
distrib_dist_adv = result[5]
distrib_dist_adv_incorrect = result[7]
distrib_dist_noise = result[10]

count = [len(distrib_dist[classe]) for classe in distrib_dist.keys()]
count_adv = [len(distrib_dist_adv[classe]) for classe in distrib_dist_adv.keys()]
count_incorrect = [len(distrib_dist_adv_incorrect[classe]) for classe in distrib_dist_adv_incorrect.keys()]
count_noise = [len(distrib_dist_noise[classe]) for classe in distrib_dist_noise.keys()]

dgms_dict = result[1]
dgms_dict_adv = result[4]
dgms_dict_adv_incorrect = result[6]
dgms_dict_noise = result[9]

c = [len(dgms_dict[classe]) for classe in dgms_dict.keys()]
c_adv = [len(dgms_dict_adv[classe]) for classe in dgms_dict_adv.keys()]
c_incorrect = [len(dgms_dict_adv_incorrect[classe]) for classe in dgms_dict_adv_incorrect.keys()]
c_noise = [len(dgms_dict_noise[classe]) for classe in dgms_dict_noise.keys()]

sum(np.asarray(c_adv) - np.asarray(c_incorrect))/(10*n)



# -----------------------
# Load reswults for plots
# -----------------------

threshold = 15000
epsilon = 0.2
noise = 0.2
import _pickle as cPickle
param = "threshold_%s_eps_%s_noise_%s/" %(threshold, epsilon, noise)    
path = "/Users/t.ricatte/dev/tda_for_adv_robustness/dict_files/"+param

with open(path+'distrib_dist.pickle', 'rb') as fp:
    distrib_dist = cPickle.load(fp)

with open(path+'distrib_dist_adv.pickle', 'rb') as fp:
    distrib_dist_adv = cPickle.load(fp)

with open(path+'distrib_dist_adv_incorrect.pickle', 'rb') as fp:
    distrib_dist_adv_incorrect = cPickle.load(fp)

with open(path+'distrib_dist_noise.pickle', 'rb') as fp:
    distrib_dist_noise = cPickle.load(fp)

for i in range(10):
    sns.distplot(distrib_dist["dist_"+str(i)+"_"+str(i)], hist=False, label="clean "+str(i))
    sns.distplot(distrib_dist_adv["dist_adv_"+str(i)], hist=False, label="adv "+str(i))
    sns.distplot(distrib_dist_adv_incorrect["dist_adv_"+str(i)], hist=False, label="adv incorrect "+str(i))
    sns.distplot(distrib_dist_noise["dist_noise_"+str(i)], hist=False, label="noise "+str(i))
    plt.show()

# ------
# Intra-class distance distribution
# ------

n=3
start_n = 60
threshold = 12000
all_class = range(10)

# n indices for every class
inds_all_class = {key: get_class_indices(key, number="all")[start_n:n+start_n] for key in all_class}

# Dict containing the persistend dgm for clean inputs (and other info),
# organized by observed class (i.e. in each class, we have the persistent dgm
# for the inputs classified correclty or not in this class)
dgms_dict = produce_dgms(net, test_set, loss_func, threshold, inds_all_class,
                 adversarial=False, epsilon=epsilon, noise=0)

# Intra-class distance distribution
distrib_dist = compute_intra_distances(dgms_dict)

#Plots
for i in all_class:
    sns.distplot(distrib_dist["dist_"+str(i)+"_"+str(i)], hist=False, label=str(i))

sns.distplot(distrib_dist["dist_1_1"], hist=False, label="1")
sns.distplot(distrib_dist["dist_7_7"], hist=False, label="7")

    
    

# ------
# Distance distribution for adv. vs clean inputs
# ------

n=3
epsilon = 0.1

# n indices for every class
inds_all_class_adv = {key: get_class_indices(key, number="all")[n+start_n:2*n+start_n] for key in all_class}

# Dict containing the persistend dgm for adv. inputs (and other info),
# organized by observed class (i.e. in each class, we have the persistent dgm
# for the inputs classified correclty or not in this class)
dgms_dict_adv = produce_dgms(net, test_set, loss_func, threshold, inds_all_class_adv,
                 adversarial=True, epsilon=epsilon, noise=0)

dgms_dict_adv_incorrect = copy.deepcopy(dgms_dict_adv)
for key in list(dgms_dict_adv_incorrect.keys()):
    for ind in list(dgms_dict_adv_incorrect[key].keys()):
        if dgms_dict_adv_incorrect[key][ind][1] == dgms_dict_adv_incorrect[key][ind][3]:
            del dgms_dict_adv_incorrect[key][ind]


# Distance distribution for adv. inputs vs clean inputs
distrib_dist_adv = compute_distances(dgms_dict, dgms_dict_adv)

for i in all_class:
    sns.distplot(distrib_dist_adv["dist_adv_"+str(i)], hist=False, label=str(i))

# Differences between the "clean" distance distrib and the adversarial one
for i in all_class:
    sns.distplot(distrib_dist["dist_"+str(i)+"_"+str(i)], hist=False, label="clean "+str(i))
    sns.distplot(distrib_dist_adv["dist_adv_"+str(i)], hist=False, label="adv "+str(i))
    plt.show()


# ------
# Distance distribution for noisy vs clean inputs
# ------

noise = 0.2

# n indices for every class
inds_all_class_noise = {key: get_class_indices(key, number="all")[2*n+start_n:3*n+start_n] for key in all_class}

# Dict containing the persistend dgm for noisy inputs (and other info),
# organized by observed class (i.e. in each class, we have the persistent dgm
# for the inputs classified correclty or not in this class)
dgms_dict_noise = produce_dgms(net, test_set, loss_func, threshold, inds_all_class_noise,
                 adversarial=False, epsilon=epsilon, noise=noise)
      
# Distance distribution for adv. inputs vs clean inputs
distrib_dist_noise = compute_distances(dgms_dict, dgms_dict_noise,
                                       adversarial=False, noisy=True)

# Plots
for i in all_class:
    sns.distplot(distrib_dist_noise["dist_noise_"+str(i)], hist=False, label=str(i))


# ------
# Differences between the three distribs
# ------

for i in all_class:
    sns.distplot(distrib_dist["dist_"+str(i)+"_"+str(i)], hist=False, label="clean "+str(i))
    sns.distplot(distrib_dist_adv["dist_adv_"+str(i)], hist=False, label="adv "+str(i))
    sns.distplot(distrib_dist_noise["dist_noise_"+str(i)], hist=False, label="noise "+str(i))
    plt.show()


count = [len(distrib_dist[classe]) for classe in distrib_dist.keys()]
count_adv = [len(distrib_dist_adv[classe]) for classe in distrib_dist_adv.keys()]
count_noise = [len(distrib_dist_noise[classe]) for classe in distrib_dist_noise.keys()]


# -----
# Save and reimport files
# -----

# Import

#with open(path+'dgms_dict.json', 'rb') as fp:
#    dgms_dict2 = pickle.load(fp)

#with open(path+'distrib_dist.json', 'r') as fp:
#    distrib_dist = json.load(fp)

#with open(path+'dgms_dict_adv.json', 'r') as fp:
#    dgms_dict_adv = json.load(fp)

#with open(path+'distrib_dist_adv.json', 'r') as fp:
#    distrib_dist_adv = json.load(fp)

#with open(path+'inds_all_class.json', 'r') as fp:
#    inds_all_class = json.load(fp)

#with open(path+'inds_all_class_adv.json', 'r') as fp:
#    inds_all_class_adv = json.load(fp)

#with open(path+'inds_all_class_noise.json', 'r') as fp:
#    inds_all_class_noise = json.load(fp)

#with open(path+'distrib_dist_noise.json', 'r') as fp:
#    distrib_dist_noise = json.load(fp)






# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

# ------ DO NOT DELETE

# Keep the inputs in the true class, not the observed class from the model
#dgms_dict = {}
#for i in inds_all_class.keys():
#    dict_temp = {}
#    for index in inds_all_class[i]:
#        dict_temp[index] = compute_persistent_dgm(net, test_set, loss_func,
#                 numero_ex=index, threshold=threshold)
#    dgms_dict["dgms_" + str(i)] = dict_temp

# ------ Dev / brouillon

#import pickle
#path = "/Users/m.goibert/Documents/Criteo/Project_2-Persistent_Homology/dict_files_output/dict_files/"
#with open(path+'dgms_dict_adv.pickle', 'rb') as fp:
#    dgms_dict_adv = pickle.load(fp)
#with open(path+'dgms_dict_noise.pickle', 'rb') as fp:
#    dgms_dict_noise = pickle.load(fp)
#with open(path+'dgms_dict.pickle', 'rb') as fp:
#    dgms_dict = pickle.load(fp)
    
#with open(path+'distrib_dist_adv.pickle', 'rb') as fp:
#    distrib_dist_adv = pickle.load(fp)
#with open(path+'distrib_dist_noise.pickle', 'rb') as fp:
#    distrib_dist_noise = pickle.load(fp)
#with open(path+'distrib_dist.pickle', 'rb') as fp:
#    distrib_dist = pickle.load(fp)
    
#with open(path+'inds_all_class_adv.pickle', 'rb') as fp:
#    inds_all_class_adv = pickle.load(fp)
#with open(path+'inds_all_class_noise.pickle', 'rb') as fp:
#    inds_all_class_noise = pickle.load(fp)
#with open(path+'inds_all_class.pickle', 'rb') as fp:
#    inds_all_class = pickle.load(fp)

