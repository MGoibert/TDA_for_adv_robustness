import copy
import os
import pathlib
from time import time

import numpy as np
import torch
from torch import nn, optim, no_grad

from tda.devices import device
from tda.models.architectures import (
    mnist_mlp,
    Architecture,
    mnist_lenet,
    svhn_lenet,
    svhn_lenet_bandw,
    svhn_lenet_bandw2,
    cifar_lenet,
    mnist_mlp_relu,
    fashion_mnist_lenet,
    fashion_mnist_mlp,
)
from tda.models.datasets import Dataset
from tda.rootpath import rootpath
from tda.tda_logging import get_logger

logger = get_logger("Models")

torch.set_default_tensor_type(torch.DoubleTensor)

pathlib.Path("/tmp/tda/trained_models").mkdir(parents=True, exist_ok=True)


def compute_val_acc(model, val_loader):
    """
        Compute the accuracy on a validation set
    """
    correct = 0
    model.eval()
    with no_grad():
        for data, target in val_loader:
            data = data.double()
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    acc = correct / len(val_loader.dataset)
    print("Val accuracy =", acc)
    return acc


def compute_test_acc(model, test_loader):
    """
    Compute the accuracy on a test set
    """
    model.eval()
    correct = 0
    with no_grad():
        for data, target in test_loader:
            data = data.double()
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    acc = correct / len(test_loader.dataset)
    print("Test accuracy =", acc)
    return acc

def go_training(model, x, y, epoch, optimizer, loss_func, train_noise=0, prune_percentile=0, first_pruned_iter=10, mask_=None):

    x = x.double()
    y = y.to(device)
    optimizer.zero_grad()

    # Normal training
    if train_noise == 0:
        y_pred = model(x)
        loss = loss_func(y_pred, y)
        loss.backward()
        optimizer.step()

    # Training with noise
    if train_noise > 0:
        logger.info(f"Training with noise...")
        if epoch >= 25: # Warm start
            x_noisy = torch.clamp(x + train_noise * torch.randn(x.size()), 0, 1).double()
            y_noisy = y_batch
            y_pred = model(x)
            y_pred_noisy = model(x_noisy)
            loss = 0.75 * loss_func(y_pred, y) + 0.25 * loss_func(y_pred_noisy, y_noisy)
            loss.backward()
            optimizer.step()
        else:
            y_pred = model(x)
            loss = loss_func(y_pred, y)
            loss.backward()
            optimizer.step()

    # Training with prune percentile
    if prune_percentile > 0:
        logger.info(f"Training with pruning...")
        for i, (name, param) in enumerate(model.named_parameters()):
            if (
                len(param.data.size()) > 1
                and epoch > first_pruned_iter
            ):
                param.data = param.data * mask_[i]
                param.grad.data = param.grad.data * mask_[i]



def train_network(
    model: Architecture,
    train_loader,
    val_loader,
    loss_func,
    num_epochs: int,
    train_noise: float = 0.0,
    prune_percentile: float = 0.0,
    tot_prune_percentile: float = 0.0,
    first_pruned_iter: int = 10,
) -> Architecture:
    """
    Helper function to train an arbitrary model
    """
    # Save model initial values
    model.epochs = num_epochs

    logger.info(f"Learnig on device {device}")

    nepochs = 0
    if prune_percentile > 0.0:
        nb_iter_prune = int(np.log(1-tot_prune_percentile)/np.log(1-prune_percentile))+1
        nepochs = num_epochs
        num_epochs = first_pruned_iter*nb_iter_prune + num_epochs
        init_weight_dict = copy.deepcopy(model.state_dict())
        logger.info(f"Training pruned NN: {nb_iter_prune} pruning iter so {num_epochs} epochs")

    if model.name in [mnist_lenet.name, fashion_mnist_lenet.name]:
        lr = 0.001
        patience = 20
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=patience, verbose=True, factor=0.5)
    elif model.name in [svhn_lenet.name, svhn_lenet_bandw.name, svhn_lenet_bandw2.name]:
        lr = 0.0008
        patience = 40
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=patience, verbose=True, factor=0.5)
    elif model.name in [mnist_mlp.name, fashion_mnist_mlp.name, mnist_mlp_relu.name]:
        lr = 0.1
        patience = 5
        optimizer = optim.SGD(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=patience, verbose=True, factor=0.5)
    elif model.name == cifar_lenet.name:
        lr = 0.0008
        patience = 50
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=patience, verbose=True, factor=0.5)

    # optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_history = []
    t = time()

    for epoch in range(num_epochs):
        logger.info(
            f"Starting epoch {epoch} ({time()-t} secs) and lr = {[param['lr'] for param in optimizer.param_groups]}"
        )
        t = time()
        model.set_train_mode()

        for x_batch, y_batch in train_loader:
            # Training !!
            if epoch == 0:
                mask_ = None
            go_training(model, x_batch, y_batch, epoch,
                optimizer, loss_func, train_noise=train_noise,
                prune_percentile=prune_percentile, first_pruned_iter=first_pruned_iter, mask_=mask_)

        model.set_eval_mode()
        for x_val, y_val in val_loader:
            y_val = y_val.to(device)
            x_val = x_val.double()
            y_val_pred = model(x_val)
            val_loss = loss_func(y_val_pred, y_val)
            logger.info(f"Validation loss = {np.around(val_loss.item(), decimals=4)}")
            loss_history.append(val_loss.item())
        if (prune_percentile == 0.0) or (epoch>first_pruned_iter*nb_iter_prune):
            scheduler.step(val_loss)
        if epoch % 10 == 9:
            logger.info(f"Val acc = {compute_val_acc(model, val_loader)}")
        if epoch > 0:
            save_pruned_model(model, current_pruned_percentile, first_pruned_iter, tot_prune_percentile, epoch, nepochs)
        if (
            (epoch+1) % first_pruned_iter == 0
            and epoch != 0
            and prune_percentile > 0.0
            and epoch < first_pruned_iter*nb_iter_prune
        ):
            logger.info(f"Pruned net epoch {epoch}")
            model, mask_ = prune_model(
                model, percentile=prune_percentile, init_weight=init_weight_dict
            )
        c = 0
        for i, p in enumerate(model.parameters()):
            c += np.count_nonzero(p.data.cpu())
        current_pruned_percentile = c/sum(p.numel() for p in model.parameters())
        logger.info(
            f"percentage non zero parameters = {current_pruned_percentile}"
        )

    return model, loss_history


def get_deep_model(
    num_epochs: int,
    dataset: Dataset,
    architecture: Architecture = mnist_mlp,
    train_noise: float = 0.0,
    prune_percentile: float = 0.0,
    tot_prune_percentile: float = 0.0,
    first_pruned_iter: int = 10,
    with_details: bool = False,
    force_retrain: bool = False,
    pretrained_pth: str = None,
) -> Architecture:

    loss_func = nn.CrossEntropyLoss()

    if pretrained_pth is not None:
        # Experimental: to help loading an existing model
        # possibly trained outside of our framework
        # Can be used with tda/models/pretrained/lenet_mnist_model.pth
        print(rootpath)
        state_dict = torch.load(
            f"{rootpath}/tda/models/pretrained/{pretrained_pth}", map_location=device
        )
        state_dict = {key.replace(".", "_"): state_dict[key] for key in state_dict}
        architecture.load_state_dict(state_dict)
        architecture.epochs = "custom"
        return architecture, loss_func

    if not os.path.exists(f"{rootpath}/trained_models"):
        os.mkdir(f"{rootpath}/trained_models")

    if train_noise > 0.0:
        nprefix = f"{train_noise}_"
    elif tot_prune_percentile > 0.0:
        nprefix = f"pruned_{np.round(1.0-tot_prune_percentile,2)}_"
    else:
        nprefix = ""

    model_filename = (
            f"{rootpath}/trained_models/{dataset.name}_"
            f"{architecture.name}_"
            f"{nprefix}"
            f"{num_epochs}_"
            f"epochs.model"
        )
    logger.info(f"Filename = {model_filename} \n")

    try:
        if force_retrain:
            raise FileNotFoundError("Force retrain")

        architecture = torch.load(model_filename, map_location=device)
        logger.info(f"Loaded successfully model from {model_filename}")
    except FileNotFoundError:
        logger.info(f"Unable to find model in {model_filename}... Retraining it...")

        x = dataset.train_dataset[0][0].to(device)
        architecture.forward(x, store_for_graph=False, output="final")
        architecture.build_matrices()
        #filename = architecture.get_model_initial_savepath()
        filename = f"{rootpath}/trained_models/{architecture.name}_{nprefix}{num_epochs}_epochs_inital.model"
        torch.save(architecture, filename)
        logger.info(f"Saved initial model in {filename}")

        # Train the NN
        train_network(
            architecture,
            dataset.train_loader,
            dataset.val_loader,
            loss_func,
            num_epochs,
            train_noise,
            prune_percentile,
            tot_prune_percentile,
            first_pruned_iter,
        )

        # Saving model
        torch.save(architecture, model_filename)

        # Compute accuracies
        val_accuracy = compute_val_acc(architecture, dataset.val_loader)
        logger.info(f"Validation accuracy = {val_accuracy}")
        test_accuracy = compute_test_acc(architecture, dataset.test_loader)
        logger.info(f"Test accuracy = {test_accuracy}")

    # Forcing eval mode just in case it was not done before
    architecture.set_eval_mode()
    architecture.is_trained = True
    architecture.epochs = num_epochs

    if with_details:
        return architecture, val_accuracy, test_accuracy

    # Build matrices
    architecture.build_matrices()

    return architecture

def save_pruned_model(architecture, current_pruned_percentile, first_pruned_iter, tot_prune_percentile, epoch, num_epochs):
    current_pruned_percentile = np.round(current_pruned_percentile,2)
    if ((tot_prune_percentile > 0.0)
        and (epoch > 0)
        and ((epoch+1) % (2*first_pruned_iter) == first_pruned_iter)):
        nprefix = f"pruned_{current_pruned_percentile}_"
        model_filename = (
            f"{rootpath}/trained_models/"
            f"{architecture.name}_"
            f"{nprefix}"
            f"{num_epochs}_"
            f"epochs.model"
        )
        logger.info(f"Save intermediate pruned model at {model_filename} \n")
        torch.save(architecture, model_filename)

def prune_model(model, percentile=0.1, init_weight=None):
    percentile = 100*percentile
    count_nonzero = 0
    count_tot = 0
    mask_dict = dict()
    num_param = len(list(model.parameters()))
    for i, (name, param) in enumerate(model.named_parameters()):
        # We only prune weight parameters (not bias)
        if len(param.data.size()) > 1:
            if i < num_param - 1:
                perc = np.percentile(abs(param.data.cpu()[param.data.cpu() != 0.0]), percentile)
            else:
                perc = np.percentile(
                    abs(param.data.cpu()[param.data.cpu() != 0.0]), percentile / 2.0
                )
            mask = torch.tensor(np.where(abs(param.data.cpu()) < perc, 0, 1)).double().to(device)
            mask_dict[i] = mask
            param.data = mask * param.data
            param.grad.data = mask * param.grad.data
            if init_weight:  # In case of reinitialization of weights
                param.data = mask * init_weight[str(name)]
            count_tot += np.prod(param.data.size())
            count_nonzero += np.count_nonzero(param.data.cpu())
    logger.info(f"Percentage pruned = {1 - count_nonzero/count_tot}")

    return model, mask_dict
