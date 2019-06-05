#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 13:32:16 2019

Author: Morgane Goibert <morgane.goibert@gmail.com>
"""

import torch
import torch.optim as optim
import torch.nn as nn
import dionysus as d
import numpy as np
import time
from tqdm import tqdm
from operator import itemgetter
from itertools import repeat

from datasets import test_set

# One-hot vector based on scalar
def one_hot(y, num_classes=None):
    if num_classes is None:
        classes, _ = y.max(0)
        num_classes = (classes.max() + 1).item()
    if y.dim() > 0:
        y_ = torch.zeros(len(y), num_classes, device=y.device)
    else:
        y_ = torch.zeros(1, num_classes)
    y_.scatter_(1, y.unsqueeze(-1), 1)
    return y_

# Cross_entropy loss (output = post-softmax output of the model, and label =
# one-hot)
def CE_loss(outputs, labels, num_classes=None):
    labels = one_hot(labels, num_classes=num_classes)
    size = len(outputs)
    if outputs[0].dim() == 0:
        for i in range(size):
            outputs[i] = outputs[i].unsqueeze(-1)
    if labels[0].dim() == 0:
        for i in range(size):
            labels[i] = labels[i].unsqueeze(-1)
    
    res = 1. / size * sum([torch.dot(torch.log(outputs[i]), labels[i])
                           for i in range(size)])
    return -res


# Train a model
def train_NN(model, train_loader, val_loader, loss_func, num_epochs):
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss_history = []
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=3, verbose=True,
            factor=0.5)
    for epoch in range(num_epochs):
        model.train()
        train_loader = tqdm(train_loader)
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.double()
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = loss_func(y_pred, y_batch)
            loss.backward()
            optimizer.step()
        model.eval()
        for x_val, y_val in val_loader:
            x_val = x_val.double()
            y_val_pred = model(x_val)
            val_loss = loss_func(y_val_pred, y_val)
            print("Validation loss = ", np.around(val_loss.item(), decimals=4))
            loss_history.append(val_loss.item())
        scheduler.step(val_loss)
    
    return model, loss_history

# Compute the accuracy on the validation set
def compute_val_acc(model, val_loader):
    correct = 0
    model.eval()
    with torch.no_grad():
        for data, target in val_loader:
            data = data.double()
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    acc = correct / len(val_loader.dataset)
    print("Val accuracy =", acc)
    return acc

# Compute the accuracy on the test set
def compute_test_acc(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.double()
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    acc = correct / len(test_loader.dataset)
    print("Test accuracy =", acc)
    return acc

# Create an adversarial example (FGMS only for now)
def adversarial_generation(model, numero_ex, epsilon=0.25, loss_func=CE_loss,
                           num_classes=10):
    x_clean = test_set[numero_ex][0].double()
    x_clean.requires_grad = True
    y_clean = torch.from_numpy(np.asarray(test_set[numero_ex][1])).unsqueeze(0)
    output = model(x_clean)
    loss = loss_func(output, y_clean, num_classes)
    model.zero_grad()
    loss.backward()
    x_adv = torch.clamp(x_clean + epsilon * x_clean.grad.data.sign(), -0.5, 0.5).double()
    
    return x_adv


# Compute the persistent diagram
def compute_persistent_dgm(model, test_set, loss_func,
                           numero_ex=0, adversarial=False, epsilon= .25,
                           threshold=0, noise=0, num_classes=None):
    
    t0 = time.time()
    # Get the parameters of the model
    # Odd items are biases, so we just keep even elements (0, 2, etc.)
    w = list(model.parameters())[::2]

    # Input
    x = test_set[numero_ex][0]
    x = x.double()

    # If we use adversarial or noisy example!
    if adversarial:
        x = adversarial_generation(model, numero_ex, epsilon,
                                   loss_func=loss_func, num_classes=num_classes)
    if noise > 0:
        x = torch.clamp(x + noise*torch.randn(x.size()), -0.5, 0.5).double()
    
    pred = model(x).argmax(dim=-1).item()
    x = x.view(-1, 28*28)
    
    # Get the neurons value for each layer (intermediate x)
    inter_x = model(x, return_intermediate=True)[1]
    
    # Get the edge weights
    # Step 1: compute the product of the weights and the intermediate x
    val = {}
    for k in range(len(w)):
        val[k] = (w[k]*inter_x[k]).detach().numpy()
    
    # Step 2: process (absolute value and rescaling)
    val = { key: 10e5*np.abs(v) for key, v in val.items() }
   
    # Create the simplicial complexes using dionysus
    # Fast implementation but "by hand"
    vec = []
    shape = np.cumsum([val[key].shape[1] for key in val.keys() ])
    shape = np.insert(shape, 0, 0)
    shape = np.insert(shape, len(shape), shape[len(shape)-1]+val[len(shape)-2].shape[0])
    
    # Old method: slower
    #for key in val.keys():
    #    for row in range(val[key].shape[0]):
    #        for col in range(val[key].shape[1]):
    #            if val[key][row,col] >= threshold:
    #                vec.append( ([row+shape[key+1], col+shape[key]],
    #                             val[key][row,col]) )
    
    #layer = {key:[] for key in range(len(val)+1)}
    for key in val.keys():
        # For vertices
        #layer[key].append(list(map( lambda t: min((i for i in t if i>=threshold), default= np.inf) , val[key].T )))
        #layer[key+1].append(list(map( lambda t: min((i for i in t if i>=threshold), default = np.inf) , val[key] )))

        # Adding the edges
        row, col = np.meshgrid(np.arange(shape[key], shape[key+1]),
                               np.arange(shape[key+1], shape[key+2]))
        table = np.vstack( (val[key].ravel(),row.ravel(),col.ravel()) ).T
        table = np.delete(table,np.where((np.asarray(list(map(itemgetter(0), table))) < threshold))[0], axis=0)
        if key == 0:
            vec = list(zip(map(list, zip( map(lambda x: int(x), map(itemgetter(1), table)),
                                         map(lambda x: int(x), map(itemgetter(2), table) ))),
                            map(itemgetter(0), table))) 
        else:
           vec = vec+list(zip(map(list, zip( map(lambda x: int(x), map(itemgetter(1), table)),
                                         map(lambda x: int(x), map(itemgetter(2), table) ))),
                            map(itemgetter(0), table))) 

    #for key in layer.keys():
    #    layer[key] = [min(x[i] for x in layer[key]) for i in range(len(layer[key][0]))]
    #    new_val = list( zip( list(map(lambda x: [x], range(shape[key],shape[key+1]))), layer[key] ) )
    #    new_val = [elem for elem in new_val if elem[1] != np.inf]
    #    vec = vec+new_val

    # Fast implementation
    # Adding the vertices
    nb_vertices = int(max([elem for array in tuple(map(itemgetter(0), vec)) for elem in array]))

    dict_vertices = {key: [] for key in range(nb_vertices+1)}
    for edge, timing in vec:
        if len(dict_vertices[edge[0]]) == 0 or  timing <= min(dict_vertices[edge[0]]):
            dict_vertices[edge[0]].append(timing)
        if len(dict_vertices[edge[1]]) == 0 or timing <= min(dict_vertices[edge[1]]):
            dict_vertices[edge[1]].append(timing)
    for vertex in dict_vertices:
        if len(dict_vertices[vertex]) > 0:
            vec.append( ([vertex], min(dict_vertices[vertex])) )
    
    # Dionysus computations (persistent diagrams)
    f = d.Filtration()
    for vertices, timing in vec:
        f.append(d.Simplex(vertices, timing))
    f.sort()
    m = d.homology_persistence(f)
    dgms = d.init_diagrams(m, f)
    
    d.plot.plot_diagram(dgms[0], show = True)
    
    t1 = time.time()
    print("Time: %s, true label = %s, pred = %s, adv = %s" %(np.round(t1 - t0, decimals=2), 
                                         test_set[numero_ex][1], pred, adversarial))
    
    return dgms[0], test_set[numero_ex][1], adversarial, pred



# Derive valid indices for a specific class
def get_class_indices(label_wanted, number=2, start_i=0, test_set=test_set):
    accept = False
    valid_numero = []
    i = start_i
    if number == "all":
        for i in range(len(test_set)):
            numero_ex = i
            y = test_set[i][1]
            if y == label_wanted:
                valid_numero.append(numero_ex)
    else:
        while accept == False:
            numero_ex = i
            y = test_set[i][1]
            if y == label_wanted:
                valid_numero.append(numero_ex)
            if len(valid_numero) == number:
                accept = True
            i = i+1      
    return valid_numero


# Compute intra-class distances distrib
def compute_intra_distances(dgms_dict):
    # Building the dictionary containing the distance distributions
    distrib_dist = {}

    # Step 1: intra-class distances distributions
    for k, key in enumerate(dgms_dict.keys()):
        dist_temp = []
        print("key =", key)
        for i, ind1 in enumerate(dgms_dict[key].keys()):
            print("ind =", ind1)
            for j in range(i+1, len(dgms_dict[key].keys())):
                ind2 = list(dgms_dict[key].keys())[j]
                dist_temp.append(
                    d.wasserstein_distance(dgms_dict[key][ind1][0],
                                            dgms_dict[key][ind2][0], q=2))
        distrib_dist["dist_"+str(k)+"_"+str(k)] = dist_temp
    return distrib_dist

# Compute adv. or noisy inputs vs clean inputs distances distrib
def compute_distances(dgms_dict, dgms_dict_perturbed, adversarial=True, noisy=False):
    add = "_adv"*adversarial + "_noise"*noisy
    distrib_dist = {}
    print("Computing distance distribution for", add, "inputs")

    for k, key in enumerate(dgms_dict.keys()):
        dist_temp = []
        print("key =", key)
        for i, ind in enumerate(dgms_dict_perturbed[key].keys()):
            print("ind"+add+" =", ind)
            for j, ind_clean in enumerate(dgms_dict[key].keys()):
                dist_temp.append(
                    d.wasserstein_distance(dgms_dict_perturbed[key][ind][0],
                                           dgms_dict[key][ind_clean][0], q=2))
        distrib_dist["dist"+add+"_"+str(k)] = dist_temp
    return distrib_dist

# Compute inter-class distribution for clean inputs.
# Careful, very long to compute
def compute_inter_distances(dgms_dict):
    distrib_dist = {}
    for class1, _ in enumerate(dgms_dict.keys()):
        key1 = "dgms_" + str(class1)
        print("key1 =", key1)
        for class2 in range(class1+1, len(dgms_dict.keys())):
            key2 = "dgms_" + str(class2)
            print("key2 =", key2)
            dist_temp = []
            for i, ind1 in enumerate(dgms_dict[key1].keys()):
                for j, ind2 in enumerate(dgms_dict[key2].keys()):
                    print("inds =", ind1, ind2)
                    dist_temp.append(
                            d.wasserstein_distance(dgms_dict[key1][ind1][0],
                                                   dgms_dict[key2][ind2][0], q=2))
        distrib_dist["dist_"+str(class1)+"_"+str(class2)] = dist_temp

    return distrib_dist


def produce_dgms(net, test_set, loss_func, threshold, inds_all_class,
                 adversarial=False, epsilon=0.25, noise=0, num_classes=None):
    
    dgms_dict = {}
    dict_temp = {}
    for i in inds_all_class.keys():
        for index in inds_all_class[i]:
            dict_temp[index] = compute_persistent_dgm(net, test_set, loss_func,
                 numero_ex=index, threshold=threshold, adversarial=adversarial,
                 epsilon=epsilon, noise=noise, num_classes=num_classes)

    for i in inds_all_class.keys():
        temp = {}
        for index in dict_temp.keys():
            if dict_temp[index][3] == i:
                temp[index] = dict_temp[index]
        dgms_dict["dgms_" + str(i)] = temp
    
    return dgms_dict

# Save results
def save_result(result, threshold, epsilon, noise):
    import os
    param = "threshold_%s_eps_%s_noise_%s/" %(threshold, epsilon, noise)    
    path = "/Users/m.goibert/Documents/Criteo/Project_2-Persistent_Homology/TDA_for_adv_robustness/dict_files/"+param
    
    if not os.path.exists(path):
        os.makedirs(path)
        
    import pickle
    import _pickle as cPickle

    # Clean input
    t0 = time.time()
    with open(path+'dgms_dict.pickle', 'wb') as fp:
        cPickle.dump(result[1], fp, protocol=pickle.HIGHEST_PROTOCOL)
    t1 = time.time()
    print("dgms_dict saved ! Time =", t1-t0)

    with open(path+'distrib_dist.pickle', 'wb') as fp:
        cPickle.dump(result[2], fp, protocol=pickle.HIGHEST_PROTOCOL)

    with open(path+'inds_all_class.pickle', 'wb') as fp:
        cPickle.dump(result[0], fp, protocol=pickle.HIGHEST_PROTOCOL)

    # Adv input
    with open(path+'dgms_dict_adv.pickle', 'wb') as fp:
        cPickle.dump(result[4], fp, protocol=pickle.HIGHEST_PROTOCOL)
    print("dgms_dict_adv saved !")

    with open(path+'distrib_dist_adv.pickle', 'wb') as fp:
        cPickle.dump(result[5], fp, protocol=pickle.HIGHEST_PROTOCOL)

    with open(path+'inds_all_class_adv.pickle', 'wb') as fp:
        cPickle.dump(result[3], fp, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(path+'dgms_dict_adv_incorrect.pickle', 'wb') as fp:
        cPickle.dump(result[6], fp, protocol=pickle.HIGHEST_PROTOCOL)
    print("dgms_dict_adv_incorrect saved !")
    
    with open(path+'distrib_dist_adv_incorrect.pickle', 'wb') as fp:
        cPickle.dump(result[7], fp, protocol=pickle.HIGHEST_PROTOCOL)

    # Noisy input
    with open(path+'dgms_dict_noise.pickle', 'wb') as fp:
        cPickle.dump(result[9], fp, protocol=pickle.HIGHEST_PROTOCOL)
    print("dgms_dict_noise saved !")

    with open(path+'inds_all_class_noise.pickle', 'wb') as fp:
        cPickle.dump(result[8], fp, protocol=pickle.HIGHEST_PROTOCOL)

    with open(path+'distrib_dist_noise.pickle', 'wb') as fp:
        cPickle.dump(result[10], fp, protocol=pickle.HIGHEST_PROTOCOL)



