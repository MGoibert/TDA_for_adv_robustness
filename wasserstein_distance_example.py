#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 14:00:50 2019

Author: Morgane Goibert <morgane.goibert@gmail.com>
"""

import dionysus as d
import numpy as np

from datasets import (test_set)
from functions import (compute_persistent_dgm, get_class_indices)
from peristent_diagram_NN import net

# ------
# Compute distances between two diagrams
# ------
    
#Get the indices for the class we want
inds_clean = get_class_indices(1, number=10)

# Distances between a clean and an adversarial input
dgms_clean = compute_persistent_dgm(net, test_set, numero_ex=inds_clean[9], adversarial=False, epsilon= .25, threshold=5000)
dgms_adv = compute_persistent_dgm(net, test_set, numero_ex=inds_clean[9], adversarial=True, epsilon= .25, threshold=5000)
wdist = d.wasserstein_distance(dgms_clean[0][0], dgms_adv[0][0], q=2)

# Distances between two clean inputs
dgms_clean2 = compute_persistent_dgm(net, test_set, numero_ex=inds_clean[5], adversarial=False, epsilon= .25, threshold=5000)
wdist_clean = d.wasserstein_distance(dgms_clean[0][0], dgms_clean2[0][0], q=2)

# Get indices for the adv predicted class
inds_adv = get_class_indices(dgms_adv[3], number=10)

# Distances between the adv exemple and the target class
dgms_compar_adv = compute_persistent_dgm(net, test_set, numero_ex=inds_adv[9], adversarial=False, epsilon= .25, threshold=5000)
wdist_compar_adv = d.wasserstein_distance(dgms_adv[0][0], dgms_compar_adv[0][0], q=2)

# Distance between the clean input and the target class input
wdist_clean_target = d.wasserstein_distance(dgms_clean[0][0], dgms_compar_adv[0][0], q=2)

print("Distance clean/adv = %s, distance clean = %s, distance adv/wrong class = %s, and distance clean/wrong = %s" 
      %(np.around(wdist, decimals=2), np.around(wdist_clean, decimals=2),
        np.around(wdist_compar_adv, decimals=2), np.around(wdist_clean_target, decimals=2)))

