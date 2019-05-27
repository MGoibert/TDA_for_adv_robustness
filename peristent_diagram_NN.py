#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 12:30:59 2019

Author: Morgane Goibert <morgane.goibert@gmail.com>
"""

from IPython.display import Image
from os import chdir
import numpy as np
import gudhi as gd
import matplotlib.pyplot as plt
import seaborn as sns
import dionysus as d
from scipy.spatial.distance import squareform

import numpy as np
import scipy.stats as stats
import scipy.sparse as sparse
import matplotlib.pyplot as plt
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


# -----
# Dyonisus library and computation

# Fill a matrix with the graph weights, and from there, compute the distance
# used to feed dyonisus. Take the inverse of the weighted matrix, replace
# the diagonal by zeros, and make the computation !
# -----


# Create a random symmetric sparse matrix
X = 10*sparse.random(30, 30, density = 0.35)
upper_X = sparse.triu(X) 
result = upper_X + upper_X.T - sparse.diags(X.diagonal())
result = result.todense()
W = 1./result
# Diagonal = 0
np.fill_diagonal(W, 0)

# Compute persistent homology
sq_dist = squareform(W)
f = d.fill_rips(sq_dist, 2, 2)
m = d.homology_persistence(f)

dgms = d.init_diagrams(m, f)
d.plot.plot_diagram(dgms[1], show = True)



# Create simplices "by hand"
simplices = [([2], 4), ([1,2], 5), ([0,2], 6),
              ([0], 1),   ([1], 2), ([0,1], 3)]
simplices = [ ([2,4],1./6), ([1,3], 1./4), ([2,3], 1./3), ([1,4],1),
             ([2], 1./6), ([4], 1./6), ([1], 1./4), ([3], 1./4)
        ]
f = d.Filtration()
for vertices, time in simplices:
    print(vertices)
    f.append(d.Simplex(vertices, time))
f.sort()
for s in f:
    print(s)
m = d.homology_persistence(f)
for i,c in enumerate(m):
    print(i, c)
dgms = d.init_diagrams(m, f)
d.plot.plot_diagram(dgms[0], show = True)
for i, dgm in enumerate(dgms):
    for pt in dgm:
        print(i, pt.birth, pt.death)



# ------------------------
# ----- Try with a true NN
# ------------------------

# MLP def
class MNISTMLP(nn.Module):
    def __init__(self):
        super(MNISTMLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 10)
        self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.soft(x)
        return x

model = MNISTMLP()

# MNIST dataset
root = './data'
trans = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
train_set = dset.MNIST(root=root, train=True,
                           transform=trans, download=True)
test_set = dset.MNIST(root=root, train=False,
                          transform=trans, download=True)

val_data = []
test = []
for i, x in enumerate(test_set):
    if i < 1000:
        val_data.append(x)
    else:
        test.append(x)

lims = -0.5, 0.5

train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                           batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test, shuffle=True,
                                          batch_size=1)
val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=len(val_data),
                                         shuffle=True)
test_loader.dataset = tuple(zip(map(lambda x: x.double(), map(itemgetter(0),
            test_loader.dataset)), map(itemgetter(1), test_loader.dataset)))

# Train the NN

num_epochs = 15
optimizer = optim.SGD(model.parameters(), lr=0.1)
loss_history = []
loss_func = nn.CrossEntropyLoss()
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
        print("Validation loss = ", val_loss)
        loss_history.append(val_loss.item())
    scheduler.step(val_loss)

# Compute val accuracy
correct = 0
model.eval()
with torch.no_grad():
    for data, target in val_loader:
        data = data.double()
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
acc = correct / len(val_loader.dataset)

# Compute test acc
model.eval()
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data = data.double()
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

acc = correct / len(test_loader.dataset)


# Paramaters
numero_ex = 5
adversarial = True
epsilon = 0.25
def compute_persistent_dgm(model, test_set, numero_ex=0, adversarial=False, epsilon= .25):

    t0 = time.time()
    # Get the parameters of the model
    # Odd items are biases
    w = list(model.parameters())
    adv_pred = -1

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
        adv_pred = model(x_adv).argmax(dim=-1).item()

    # Induce model parameters
    x = test_set[numero_ex][0]
    # x[1]
    # If we use adversarial example !
    if adversarial:
        x = x_adv
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
    val0, val2, val4 = 1000*np.abs(val0), 1000*np.abs(val2), 1000*np.abs(val4)
    val0, val2, val4 = np.around(val0, decimals=7), np.around(val2, decimals=7), np.around(val4, decimals=7)
    
    # Fast implementation but "by hand"
    vec = []

    for row in range(val0.shape[0]):
        for col in range(val0.shape[1]):
            vec.append( ([row+val0.shape[1], col], val0[row,col]) )
        
    for row in range(val2.shape[0]):
        for col in range(val2.shape[1]):
            vec.append( ([row+val0.shape[1]+val2.shape[1], col+val0.shape[1]], val2[row,col]) )

    for row in range(val4.shape[0]):
        for col in range(val4.shape[1]):
            vec.append( ([row+val0.shape[1]+val2.shape[1]+val4.shape[1], 
                      col+val0.shape[1]+val2.shape[1]], val4[row,col]) )
    
    # Fast implementation
    nb_vertices = max([elem for array in tuple(map(itemgetter(0), vec)) for elem in array])

    dict_vertices = {key: [] for key in range(nb_vertices+1)}
    for edge, timing in vec:
        dict_vertices[edge[0]].append(timing)
        dict_vertices[edge[1]].append(timing)
    for vertex in dict_vertices:
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
        adv_label = " (adv label = " + str(adv_pred)+ ")"
    print("Time: %s and true label = %s %s" %(np.round(t1 - t0, decimals=2), 
                                         test_set[numero_ex][1], adv_label))


    #for pt in dgms[0]:
    #    print(0, pt.birth, pt.death)
    
    return dgms, test_set[numero_ex][1], adversarial, adv_pred


dgms1 = compute_persistent_dgm(model, test_set, numero_ex=0, adversarial=False, epsilon= .25)



# ------
# Compute distances between two diagrams
# ------

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
        
#Get the indices for the class we want
inds_clean = get_class_indices(6)

# Distances between a clean and an adversarial input
dgms_clean = compute_persistent_dgm(model, test_set, numero_ex=inds_clean[0], adversarial=False, epsilon= .25)
dgms_adv = compute_persistent_dgm(model, test_set, numero_ex=inds_clean[0], adversarial=True, epsilon= .25)
wdist = d.wasserstein_distance(dgms_clean[0][0], dgms_adv[0][0], q=2)

# Distances between two clean inputs
dgms_clean2 = compute_persistent_dgm(model, test_set, numero_ex=inds_clean[1], adversarial=False, epsilon= .25)
wdist_clean = d.wasserstein_distance(dgms_clean[0][0], dgms_clean2[0][0], q=2)

# Get indices for the adv predicted class
inds_adv = get_class_indices(dgms_adv[3])

# Distances between the adv exemple and the target class
dgms_compar_adv = compute_persistent_dgm(model, test_set, numero_ex=inds_adv[0], adversarial=False, epsilon= .25)
wdist_compar_adv = d.wasserstein_distance(dgms_adv[0][0], dgms_compar_adv[0][0], q=2)

print("Distance clean/adv = %s, distance clean = %s, distance adv/wrong class = %s" 
      %(np.around(wdist, decimals=2), np.around(wdist_clean, decimals=2),
        np.around(wdist_compar_adv, decimals=2)))



# ------
# Compute intra-class distribution of distances
# ------

# We want to display the distance distribution of a class, and the distance
# distribution of an av. input from that class (but misclassified into another
# class by the model) to the data from the original class.
inds_class = get_class_indices(0, number="all")
inds_class = inds_class[:100]
dgms_dict = {key: [] for key in inds_class}

for index in inds_class:
    dgms_dict[index].append( compute_persistent_dgm(model, test_set, numero_ex=index)[0] )

dist_vec = []
for i, ind1 in enumerate(inds_class):
    print(i)
    for j in range(i+1, len(inds_class)):
        dist_vec.append( d.wasserstein_distance(dgms_dict[ind1][0][0], dgms_dict[inds_class[j]][0][0], q=2) )

sns.distplot(dist_vec)


# Compute the distance between an adversarial and the class
dgms_adv_dist = compute_persistent_dgm(model, test_set, numero_ex=inds_class[101], adversarial=True, epsilon= .25)
wrong_class = dgms_adv_dist[3]

dist_vec_adv = []
for i, ind1 in enumerate(inds_class):
    print(i)
    dist_vec_adv.append( d.wasserstein_distance(dgms_dict[ind1][0][0], dgms_adv_dist[0][0], q=2) )

sns.distplot(dist_vec_adv)

# On the same plot
sns.distplot(dist_vec, hist=False)
sns.distplot(dist_vec_adv, hist=False)
sns.plt.show()


# And compute the distance with the wrong class
# Wa want to display the distance distribution for class "wrong" (i.e. the
# wrong predicted class for the adv. input), and the distance distribution
# of the adv. input and this "wrong" class (thus same prediction class)
inds_class_wrong = get_class_indices(wrong_class, number="all")
inds_class_wrong = inds_class_wrong[:100]
dgms_dict_wrong = {key: [] for key in inds_class_wrong}

for index_wrong in inds_class_wrong:
    dgms_dict_wrong[index_wrong].append( compute_persistent_dgm(model, test_set, numero_ex=index_wrong)[0] )

dist_vec_wrong = []
for i, ind1 in enumerate(inds_class_wrong):
    print(i)
    for j in range(i+1, len(inds_class_wrong)):
        dist_vec_wrong.append( d.wasserstein_distance(dgms_dict_wrong[ind1][0][0], dgms_dict_wrong[inds_class_wrong[j]][0][0], q=2) )

sns.distplot(dist_vec_wrong)

dist_vec_adv_wrong = []
for i, ind1 in enumerate(inds_class_wrong):
    print(i)
    dist_vec_adv_wrong.append( d.wasserstein_distance(dgms_dict_wrong[ind1][0][0], dgms_adv_dist[0][0], q=2) )

sns.distplot(dist_vec_adv_wrong)

# On the same plot
sns.distplot(dist_vec_wrong, hist=False)
sns.distplot(dist_vec_adv_wrong, hist=False)
sns.plt.show()


# All plots together
sns.distplot(dist_vec, hist=False, label="clean")
sns.distplot(dist_vec_adv, hist=False, label="clean/adv")
sns.distplot(dist_vec_wrong, hist=False, label="wrong")
sns.distplot(dist_vec_adv_wrong, hist=False, label="wrong/adv")
sns.plt.show()


# We want to display the distance between the orginal class and the "wrong" class
dist_vec_origin_wrong = []
for i, ind1 in enumerate(inds_class[0:25]):
    print(i)
    for j, ind2 in enumerate(inds_class_wrong[0:25]):
        dist_vec_origin_wrong.append( d.wasserstein_distance(dgms_dict[ind1][0][0], dgms_dict_wrong[ind2][0][0], q=2) )

sns.distplot(dist_vec_origin_wrong, hist=False, label="origin/wrong")
sns.distplot(dist_vec, hist=False, label="clean")
sns.distplot(dist_vec_adv, hist=False, label="clean/adv")
sns.distplot(dist_vec_wrong, hist=False, label="wrong")
sns.distplot(dist_vec_adv_wrong, hist=False, label="wrong/adv")
sns.plt.show()




# -----
# Other way, !! Not valid !!
# -----

matrix0 = np.concatenate(
        (np.zeros((784,2050)), block_diag(val0, val2, val4, np.concatenate((val6, np.zeros((10,10))), axis=1))), 
        axis=0)
matrix1 = np.tril(matrix0)
matrix_fin = matrix1 + matrix1.T
matrix_fin[matrix_fin > 0].min()
matrix_fin[matrix_fin > 0].max()
matrix_fin2 = np.around(matrix_fin, decimals=3)
matrix_fin2[matrix_fin2==0] = inf


# Distance matrix = inverse weight matrix
W = 1./matrix_fin2
W = np.abs(W)
np.max(W[W<inf])
np.min(W[W>0])
# Diagonal = 0
W[W==0]=10e4
np.fill_diagonal(W, 0)
W = np.around(W)

# Compute persistent homology
sq_dist = squareform(W)
f = d.fill_rips(sq_dist, 1, 2000)
m = d.homology_persistence(f)

dgms = d.init_diagrams(m, f)
d.plot.plot_diagram(dgms[0], show = True)






# ------ Dev / brouillon

