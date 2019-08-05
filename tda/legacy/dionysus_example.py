#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 13:21:34 2019

Author: Morgane Goibert <morgane.goibert@gmail.com>
"""

# -----
# Dyonisus library and computation

# Fill a matrix with the graph weights, and from there, compute the distance
# used to feed dyonisus. Take the inverse of the weighted matrix, replace
# the diagonal by zeros, and make the computation !
# -----

import seaborn as sns
import dionysus as d
import scipy.sparse as sparse
import numpy as np
from scipy.spatial.distance import squareform
sns.set()

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



# Create simplices "by hand": a small example
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