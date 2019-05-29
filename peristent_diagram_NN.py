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
                       compute_persistent_dgm, get_class_indices)



# ------------------------
# ----- Try with a true NN
# ------------------------

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
# Compute all distribution of distances
# ------

n=3
all_class = range(10)

# n indices for every class
inds_all_class = {key: get_class_indices(key, number="all")[:n] for key in all_class}

# Dict containing n dgms for each class (key = class, "dgms_i"; value = dgms for
# each index for each class)
dgms_dict = {}
for i in inds_all_class.keys():
    dict_temp = {}
    for index in inds_all_class[i]:
        dict_temp[index] = compute_persistent_dgm(net, test_set, loss_func,
                 numero_ex=index, threshold=5000)
    dgms_dict["dgms_" + str(i)] = dict_temp

# Building the dictionary containing the distance distributions
distrib_dist = {}

# Step 1: intra-class distances distributions
for k, key in enumerate(dgms_dict.keys()):
    dist_temp = []
    print("key =", key)
    for i, ind1 in enumerate(dgms_dict[key].keys()):
        print("ind1 =", ind1)
        for j in range(i+1, len(dgms_dict[key].keys())):
            ind2 = list(dgms_dict[key].keys())[j]
            print("ind2 =", ind2)
            dist_temp.append(
                    d.wasserstein_distance(dgms_dict[key][ind1][0][0],
                                           dgms_dict[key][ind2][0][0], q=2) )
    distrib_dist["dist_"+str(k)+"_"+str(k)] = dist_temp

# Step 2: inter-class distribution
#for class1, _ in enumerate(all_class):
#    key1 = "dgms_" + str(class1)
#    print("key1 =", key1)
#    for class2 in range(class1+1, len(all_class)):
#        key2 = "dgms_" + str(class2)
#        print("key2 =", key2)
#        dist_temp = []
#        for i, ind1 in enumerate(dgms_dict[key1].keys()):
#            for j, ind2 in enumerate(dgms_dict[key2].keys()):
#                print("inds =", ind1, ind2)
#                dist_temp.append( d.wasserstein_distance(dgms_dict[key1][ind1][0][0], dgms_dict[key2][ind2][0][0], q=2) )
#        distrib_dist["dist_"+str(class1)+"_"+str(class2)] = dist_temp

# Plots
for i in all_class:
    sns.distplot(distrib_dist["dist_"+str(i)+"_"+str(i)], hist=False, label=str(i))

#for i in all_class:
#    sns.distplot(distrib_dist["dist_0_"+str(i)], hist=False, label="0 and "+str(i))

sns.distplot(distrib_dist["dist_1_1"], hist=False, label="1")
sns.distplot(distrib_dist["dist_7_7"], hist=False, label="7")
#sns.distplot(distrib_dist["dist_1_7"], hist=False, label="1 and 7")




# ------
# Compute persistent diagrams and distances for several adversarial examples
# ------

epsilon = 0.25

# n indices for every class
inds_all_class_adv = {key: get_class_indices(key, number="all")[n:2*n] for key in all_class}

# Dict containing n adv. dgms indexed by the target "wrong" class the model
# (badly) predicted
dgms_dict_adv = {}
dict_temp = {}
for i in inds_all_class_adv.keys():
    for index in inds_all_class_adv[i]:
        dict_temp[index] = compute_persistent_dgm(net, test_set, loss_func,
                 numero_ex=index, threshold=5000, adversarial=True)

for i in inds_all_class_adv.keys():
    temp = {}
    for index in dict_temp.keys():
        if dict_temp[index][3] == i:
            temp[index] = dict_temp[index]
    dgms_dict_adv["dgms_" + str(i)] = temp
        
# Building the dictionary containing the distance distributions between adv
# examples and the target "wrong" class.
distrib_dist_adv = {}

for k, key in enumerate(dgms_dict.keys()):
    dist_temp = []
    print("key =", key)
    for i_adv, ind_adv in enumerate(dgms_dict_adv[key].keys()):
        print("ind_adv =", ind_adv)
        for j, ind_clean in enumerate(dgms_dict[key].keys()):
            print("ind clean =", ind_clean)
            dist_temp.append(
                    d.wasserstein_distance(dgms_dict_adv[key][ind_adv][0][0],
                                           dgms_dict[key][ind_clean][0][0], q=2))
    distrib_dist_adv["dist_adv_"+str(k)] = dist_temp

for i in all_class:
    sns.distplot(distrib_dist_adv["dist_adv_"+str(i)], hist=False, label=str(i))



# ------
# Differences between the "clean" distance distrib and the adversarial one
# ------

for i in all_class:
    sns.distplot(distrib_dist["dist_"+str(i)+"_"+str(i)], hist=False, label="clean "+str(i))
    sns.distplot(distrib_dist_adv["dist_adv_"+str(i)], hist=False, label="adv "+str(i))
    plt.show()


# ------
# Differences between the "clean" distance distrib and noisy distrib
# ------

noise = 0.25

# n indices for every class
inds_all_class_noise = {key: get_class_indices(key, number="all")[2*n:3*n] for key in all_class}

# Dict containing n noisy dgms indexed by the target "wrong" class the model
# (badly) predicted
dgms_dict_noise = {}
dict_temp = {}
for i in inds_all_class_noise.keys():
    for index in inds_all_class_noise[i]:
        dict_temp[index] = compute_persistent_dgm(net, test_set, loss_func,
                 numero_ex=index, threshold=5000, noise=noise)

for i in inds_all_class_noise.keys():
    temp = {}
    for index in dict_temp.keys():
        if dict_temp[index][3] == i:
            temp[index] = dict_temp[index]
    dgms_dict_noise["dgms_" + str(i)] = temp
        

# Building the dictionary containing the distance distributions between noisy
# examples and the target "wrong" class.
distrib_dist_noise = {}

for k, key in enumerate(dgms_dict.keys()):
    dist_temp = []
    print("key =", key)
    for i_noise, ind_noise in enumerate(dgms_dict_noise[key].keys()):
        print("ind_noise =", ind_noise)
        for j, ind_clean in enumerate(dgms_dict[key].keys()):
            print("ind clean =", ind_clean)
            dist_temp.append(
                    d.wasserstein_distance(dgms_dict_noise[key][ind_noise][0][0],
                                           dgms_dict[key][ind_clean][0][0], q=2))
    distrib_dist_noise["dist_noise_"+str(k)] = dist_temp

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



# ------ Dev / brouillon

a = {0: [1,2,3], 1: [10,20,30]}
b = {}
for i in a.keys():
    b["dgms_" + str(i)] = "test"+str(i)
