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

from datasets import test_set

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


# Compute the persistent diagram
def compute_persistent_dgm(model, test_set, loss_func,
                           numero_ex=0, adversarial=False, epsilon= .25,
                           threshold=0, noise=0):

    t0 = time.time()
    # Get the parameters of the model
    # Odd items are biases
    w = list(model.parameters())

    # Create an adversarial example
    if adversarial:
        x_clean = test_set[numero_ex][0]
        x_clean = x_clean.double()
        y_clean = torch.from_numpy(np.asarray(test_set[numero_ex][1])).unsqueeze(0)

        x_clean.requires_grad = True
        output = model(x_clean)
        loss = loss_func(output, y_clean)
        model.zero_grad()
        loss.backward()
        x_adv = torch.clamp(x_clean + epsilon * x_clean.grad.data.sign(), -0.5, 0.5)
        pred = model(x_adv).argmax(dim=-1).item()

    # Induce model parameters
    x = test_set[numero_ex][0]
    x = x.double()
    # x[1]
    # If we use adversarial example !
    if adversarial:
        x = x_adv
    if noise > 0:
        x = torch.clamp(x + noise*torch.randn(x.size()), -0.5, 0.5)
    pred = model(x).argmax(dim=-1).item()
    x = x.view(-1, 28*28)
    x.size()
    
    w0 = w[0]
    w0.size()
    val0 = w0*x.double()
    val0.size()
    res0 = (torch.mm(w0, x.double().transpose(0,1)) + w[1].view(-1,1)).transpose(0,1)
    res0.size()

    w2 = w[2]
    w2.size()
    val2 = w2*res0
    val2.size()
    res2 = (torch.mm(w2, res0.transpose(0,1)) + w[3].view(-1,1)).transpose(0,1)
    res2.size()
    
    w4 = w[4]
    w4.size()
    val4 = w4*res2
    val4.size()
    res4 = (torch.mm(w4, res2.transpose(0,1)) + w[5].view(-1,1)).transpose(0,1)
    res4.size()

    # Create the final weight matrix using blocks
    val0, val2, val4= val0.detach().numpy(), val2.detach().numpy(), val4.detach().numpy()
    val0, val2, val4 = 10e5*np.abs(val0), 10e5*np.abs(val2), 10e5*np.abs(val4)
    val0, val2, val4 = np.around(val0, decimals=2), np.around(val2, decimals=2), np.around(val4, decimals=2)
    
    # Fast implementation but "by hand"
    vec = []
    
    # Adding the edges
    # If the edge value is too small (< threshold), then there is no edge.
    # 0 is not a possible value for the model parameter, but a very small value
    # can be considered as a 0.
    for row in range(val0.shape[0]):
        for col in range(val0.shape[1]):
            if val0[row,col] >= threshold:
                vec.append( ([row+val0.shape[1], col], val0[row,col]) )
        
    for row in range(val2.shape[0]):
        for col in range(val2.shape[1]):
            if val2[row,col] >= threshold:
                vec.append( ([row+val0.shape[1]+val2.shape[1], col+val0.shape[1]], val2[row,col]) )

    for row in range(val4.shape[0]):
        for col in range(val4.shape[1]):
            if val4[row,col] >= threshold:
                vec.append( ([row+val0.shape[1]+val2.shape[1]+val4.shape[1], 
                      col+val0.shape[1]+val2.shape[1]], val4[row,col]) )
    
    # Fast implementation
    # Adding the vertices
    nb_vertices = max([elem for array in tuple(map(itemgetter(0), vec)) for elem in array])

    dict_vertices = {key: [] for key in range(nb_vertices+1)}
    for edge, timing in vec:
        dict_vertices[edge[0]].append(timing)
        dict_vertices[edge[1]].append(timing)
    for vertex in dict_vertices:
        if len(dict_vertices[vertex]) > 0:
            vec.append( ([vertex], min(dict_vertices[vertex])) )
    
    
    f = d.Filtration()
    for vertices, timing in vec:
        f.append(d.Simplex(vertices, timing))
    f.sort()
    m = d.homology_persistence(f)
    #for i,c in enumerate(m):
    #    print(i, c)
    dgms = d.init_diagrams(m, f)
    d.plot.plot_diagram(dgms[0], show = True)
        
    t1 = time.time()
    adv_label = ""
    if adversarial:
        adv_label = " (adv label = " + str(pred)+ ")"
    print("Time: %s and true label = %s %s" %(np.round(t1 - t0, decimals=2), 
                                         test_set[numero_ex][1], adv_label))


    #for pt in dgms[0]:
    #    print(0, pt.birth, pt.death)
    
    return dgms, test_set[numero_ex][1], adversarial, pred



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



