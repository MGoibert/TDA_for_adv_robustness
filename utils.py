#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 15:00:30 2019

Author: Morgane Goibert <morgane.goibert@gmail.com>
"""

import sys
import logging
import argparse

# config
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

def parse_cmdline_args():
    
    parser = argparse.ArgumentParser(
        description="Run TDA experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--dataset', type=str, choices=["MNIST"],
        default="MNIST", help="Currently not implemented")
    parser.add_argument(
        '--model', type=str, choices=["MLP"],
        default="MLP", help="Currently not implemented")
    parser.add_argument(
        '--num_epochs', type=int, default=20,
        help="number of passes to make over data")
    parser.add_argument(
        '--epsilon', type=float, default=0.25,
        help="adversarial perturbation")
    parser.add_argument(
        '--noise', type=float, default=0.25,
        help="noise perturbation")
    parser.add_argument(
        '--threshold', type=float, default=5000,
        help="Threshold for considering an edge value = 0")
    parser.add_argument(
        '--num_computation', type=int, default=5,
        help="number of induced graphs and persistent dgms to compute")
    parser.add_argument(
        '--save', action="store_true",
        help="if mentionned, save the dgms, distance distributions and indices")
    
    return parser.parse_args()