#!/usr/bin/env python
# coding: utf-8

import argparse
import logging
import os
import time

from r3d3.experiment_db import ExperimentDB

from tda.models import mnist_mlp, Dataset, get_deep_model
from tda.models.architectures import get_architecture, Architecture
from tda.rootpath import db_path

start_time = time.time()

plot_path = f"{os.path.dirname(os.path.realpath(__file__))}/plots"
if not os.path.exists(plot_path):
    os.mkdir(plot_path)

my_db = ExperimentDB(db_path=db_path)

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_id', type=int, default=-1)
parser.add_argument('--run_id', type=int, default=-1)

parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--dataset', type=str, default="MNIST")
parser.add_argument('--architecture', type=str, default=mnist_mlp.name)
parser.add_argument('--dataset_size', type=int, default=100)
parser.add_argument('--attack_type', type=str, default="FGSM")
parser.add_argument('--epsilon', type=float, default=0.02)
parser.add_argument('--noise', type=float, default=0.0)

args, _ = parser.parse_known_args()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(f"[{args.experiment_id}_{args.run_id}]")

for arg in vars(args):
    logger.info(f"{arg} => {getattr(args, arg)}")

dataset = Dataset(name=args.dataset)

logger.info(f"Getting deep model...")
model, loss_func = get_deep_model(
    num_epochs=args.epochs,
    dataset=dataset,
    architecture=get_architecture(args.architecture),
    train_noise=0.0
)

archi: Architecture = model

logger.info(f"Got deep model...")

my_db.update_experiment(
    experiment_id=args.experiment_id,
    run_id=args.run_id,
    metrics={
        "time": time.time()-start_time
    }
)
