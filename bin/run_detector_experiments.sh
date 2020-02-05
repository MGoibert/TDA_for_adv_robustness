#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=""

r3d3-xp --experiment_file tda/experiments/ocsvm_detector/main_mnist_mlp.py
r3d3-xp --experiment_file tda/experiments/ocsvm_detector/main_mnist_lenet.py
r3d3-xp --experiment_file tda/experiments/ocsvm_detector/main_svhn.py

r3d3-xp --experiment_file tda/experiments/lid/lid_mnist_mlp.py
r3d3-xp --experiment_file tda/experiments/lid/lid_mnist.py
r3d3-xp --experiment_file tda/experiments/lid/lid_svhn.py

r3d3-xp --experiment_file tda/experiments/mahalanobis/mahalanobis_mnist_mlp.py
r3d3-xp --experiment_file tda/experiments/mahalanobis/mahalanobis_mnist.py
r3d3-xp --experiment_file tda/experiments/mahalanobis/mahalanobis_svhn.py