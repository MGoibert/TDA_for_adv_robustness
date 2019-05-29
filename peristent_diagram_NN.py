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
from scipy.spatial.distance import squareform

import scipy.stats as stats
import scipy.sparse as sparse
from numpy import inf
from scipy.linalg import block_diag

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
from operator import itemgetter
from tqdm import tqdm
import time
torch.set_default_tensor_type(torch.DoubleTensor)
sns.set()

# Files from the project to import
from NN_architecture import MNISTMLP
from datasets import (train_loader_MNIST, test_loader_MNIST, val_loader_MNIST,
                      test_set)
from functions import (train_NN, compute_val_acc, compute_test_acc,
                       compute_persistent_dgm, get_class_indices,
                       compute_intra_distances, compute_distances,
                       produce_dgms)
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
num_epochs = 1
loss_func = nn.CrossEntropyLoss()
net = train_NN(model, train_loader, val_loader, loss_func, num_epochs)[0]

# Compute accuarcies
compute_val_acc(model, val_loader)
compute_test_acc(model, test_loader)

# Test how it is working
dgms1 = compute_persistent_dgm(net, test_set, loss_func, numero_ex=0,
                               adversarial=False, epsilon= .25, noise=0.25,
                               threshold=5000)


# ------
# Intra-class distance distribution
# ------

n=3
threshold = 5000
all_class = range(10)

# n indices for every class
inds_all_class = {key: get_class_indices(key, number="all")[:n] for key in all_class}

# Dict containing the persistend dgm for clean inputs (and other info),
# organized by observed class (i.e. in each class, we have the persistent dgm
# for the inputs classified correclty or not in this class)
dgms_dict = produce_dgms(net, test_set, loss_func, threshold, inds_all_class,
                 adversarial=False, noise=0)

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

epsilon = 0.25

# n indices for every class
inds_all_class_adv = {key: get_class_indices(key, number="all")[n:2*n] for key in all_class}

# Dict containing the persistend dgm for adv. inputs (and other info),
# organized by observed class (i.e. in each class, we have the persistent dgm
# for the inputs classified correclty or not in this class)
dgms_dict_adv = produce_dgms(net, test_set, loss_func, threshold, inds_all_class,
                 adversarial=True, noise=0)

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

noise = 0.25

# n indices for every class
inds_all_class_noise = {key: get_class_indices(key, number="all")[2*n:3*n] for key in all_class}

# Dict containing the persistend dgm for noisy inputs (and other info),
# organized by observed class (i.e. in each class, we have the persistent dgm
# for the inputs classified correclty or not in this class)
dgms_dict_noise = produce_dgms(net, test_set, loss_func, threshold, inds_all_class,
                 adversarial=False, noise=noise)
      
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


# -----
# Save and reimport files
# -----

# Save

if save:    
    path = "/Users/m.goibert/Documents/Criteo/Project_2-Persistent_Homology/TDA_for_adv_robustness/dict_files/"
    import pickle

    # Clean input
    with open(path+'dgms_dict.pickle', 'wb') as fp:
        pickle.dump(dgms_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
    print("dgms_dict saved !")

    with open(path+'distrib_dist.pickle', 'wb') as fp:
        pickle.dump(distrib_dist, fp, protocol=pickle.HIGHEST_PROTOCOL)

    with open(path+'inds_all_class.pickle', 'wb') as fp:
        pickle.dump(inds_all_class, fp, protocol=pickle.HIGHEST_PROTOCOL)

    # Adv input
    with open(path+'dgms_dict_adv.pickle', 'wb') as fp:
        pickle.dump(dgms_dict_adv, fp, protocol=pickle.HIGHEST_PROTOCOL)
    print("dgms_dict_adv saved !")

    with open(path+'distrib_dist_adv.pickle', 'wb') as fp:
        pickle.dump(distrib_dist_adv, fp, protocol=pickle.HIGHEST_PROTOCOL)

    with open(path+'inds_all_class_adv.pickle', 'wb') as fp:
        pickle.dump(inds_all_class_adv, fp, protocol=pickle.HIGHEST_PROTOCOL)

    # Noisy input
    with open(path+'dgms_dict_noise.pickle', 'wb') as fp:
        pickle.dump(dgms_dict_noise, fp, protocol=pickle.HIGHEST_PROTOCOL)
    print("dgms_dict_noise saved !")

    with open(path+'inds_all_class_noise.pickle', 'wb') as fp:
        pickle.dump(inds_all_class_noise, fp, protocol=pickle.HIGHEST_PROTOCOL)

    with open(path+'distrib_dist_noise.pickle', 'wb') as fp:
        pickle.dump(distrib_dist_noise, fp, protocol=pickle.HIGHEST_PROTOCOL)



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

adversarial = False
noisy = True
add = "_adv"*adversarial + "_noise"*noisy
"dist"+add+"_"






