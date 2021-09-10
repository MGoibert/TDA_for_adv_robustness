import copy
import os
import pathlib
from time import time

import numpy as np
import torch
import mlflow
from torch import nn, optim, no_grad
from torch.optim.lr_scheduler import _LRScheduler
from typing import List, Optional

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
    cifar_resnet_1,
    svhn_resnet_1,
    toy_mlp,
    toy_mlp2,
    toy_mlp3,
    toy_mlp4,
    toy_viz,
    toy_viz2,
    efficientnet,
)
from tda.dataset.datasets import Dataset
from tda.models.layers import ConvLayer, LinearLayer
from tda.rootpath import rootpath
from tda.tda_logging import get_logger
from tda.precision import default_tensor_type

torch.set_default_tensor_type(default_tensor_type)

logger = get_logger("Models")

mlflow.set_tracking_uri("https://mlflow.par.prod.crto.in")
mlflow.set_experiment("tda_adv_detection")

pathlib.Path("/tmp/tda/trained_models").mkdir(parents=True, exist_ok=True)


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler):
        self.multiplier = multiplier
        if self.multiplier <= 1.0:
            raise ValueError("multiplier should be greater than 1.")
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        return [
            base_lr
            / self.multiplier
            * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=0):
        if epoch > self.total_epoch:
            self.after_scheduler.step(epoch - self.total_epoch)
        else:
            super(GradualWarmupScheduler, self).step(epoch)


def compute_val_acc(model, val_loader):
    """
        Compute the accuracy on a validation set
    """
    correct = 0
    model.eval()
    with no_grad():
        for data, target in val_loader:
            data = data.type(default_tensor_type)
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
            data = data.type(default_tensor_type)
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    acc = correct / len(test_loader.dataset)
    print("Test accuracy =", acc)
    return acc


def go_training(
    model,
    x,
    y,
    epoch,
    optimizer,
    loss_func,
    train_noise=0,
    prune_percentile=0,
    first_pruned_iter=10,
    mask_=None,
):

    x = x.type(default_tensor_type)
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
        if epoch >= 25:  # Warm start
            x_noisy = torch.clamp(x + train_noise * torch.randn(x.size()), 0, 1).type(
                default_tensor_type
            )
            y_pred = model(x)
            y_pred_noisy = model(x_noisy)
            loss = 0.75 * loss_func(y_pred, y) + 0.25 * loss_func(y_pred_noisy, y)
            loss.backward()
            optimizer.step()
        else:
            y_pred = model(x)
            loss = loss_func(y_pred, y)
            loss.backward()
            optimizer.step()

    # Training with prune percentile
    if prune_percentile > 0 and mask_ != None:
        for i, (name, param) in enumerate(model.named_parameters()):
            if len(param.data.size()) > 1:
                param.data = param.data * mask_[i]
                param.grad.data = param.grad.data * mask_[i]
    
    return loss.item()


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
    # Save initial model
    torch.save(model, model.get_model_savepath(initial=True))
    mlflow.log_artifact(model.get_model_savepath(initial=True), "models")

    # Save model initial values
    model.epochs = num_epochs
    model.to_device(device)

    def init_weights(mod):
        if isinstance(mod, torch.nn.Conv2d) or isinstance(mod, torch.nn.Linear):
            torch.nn.init.xavier_uniform(mod.weight)

    for layer in model.layers:
        if isinstance(layer, ConvLayer) or isinstance(layer, LinearLayer):
            layer.func.apply(init_weights)

    logger.info(f"Learnig on device {device}")

    custom_scheduler_step = None

    nepochs = 0
    if prune_percentile > 0.0:
        nb_iter_prune = (
            int(np.log(1 - tot_prune_percentile) / np.log(1 - prune_percentile)) + 1
        )
        nb_iter_prune = 10
        nepochs = num_epochs
        num_epochs = first_pruned_iter * nb_iter_prune + num_epochs
        # logger.info(f"The initial model = {architecture.get_model_savepath(initial=True)}")
        modelinit = torch.load(
            f"{rootpath}/trained_models/cifar_resnet_1_e_99_init.model",
            map_location=device,
        )
        init_weight_dict = copy.deepcopy(modelinit.state_dict())
        logger.info(
            f"Training pruned NN: {nb_iter_prune} pruning iter so {num_epochs} epochs with {prune_percentile} prune %"
        )
        logger.info(f"Loading last pruned model")
        del model
        model = torch.load(
            f"{rootpath}/trained_models/cifar_resnet_1_e_99_p_0.24.model",
            map_location=device,
        )
        model.epochs = 99
        model.to_device(device)
        model.tot_prune_percentile = 0.76
    else:
        init_weight_dict = None

    if model.name in [mnist_lenet.name, fashion_mnist_lenet.name, toy_viz.name]:
        lr = 0.0001 #0.001
        patience = 40 #20
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=patience, verbose=True, factor=0.5
        )
    elif model.name in [svhn_lenet.name, svhn_lenet_bandw.name, svhn_lenet_bandw2.name]:
        lr = 0.0008
        patience = 40
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=patience, verbose=True, factor=0.5
        )
    elif model.name in [mnist_mlp.name, fashion_mnist_mlp.name, mnist_mlp_relu.name]:
        lr = 0.1
        patience = 5
        optimizer = optim.SGD(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=patience, verbose=True, factor=0.5
        )
    elif model.name in [cifar_lenet.name]:
        lr = 0.0008
        patience = 50
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=patience, verbose=True, factor=0.5
        )
    elif model.name in [cifar_resnet_1.name, svhn_resnet_1.name]:

        def lr(epoch):
            if epoch > 50:
                return lr(epoch % 50)
            elif epoch < 20:
                a = (0.12 - 0.008) / 20
                b = 0.008
            elif epoch < 40:
                a = (0.008 - 0.12) / (40 - 20)
                b = 0.12 - a * 20
            else:
                a = (0.0008 - 0.008) / (50 - 40)
                b = 0.008 - a * 40
            return a * epoch + b

        scheduler = None

        def custom_scheduler_step(optimizer, epoch):
            new_lr = lr(epoch)
            for param_group in optimizer.param_groups:
                param_group["lr"] = new_lr

        optimizer = optim.SGD(
            model.parameters(), lr=lr(0), weight_decay=0.0005, momentum=0.9
        )
    elif model.name in [toy_mlp.name, toy_mlp2.name, toy_mlp3, toy_mlp4, toy_viz2.name]:
        lr = 1
        patience = 5
        optimizer = optim.SGD(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=patience, verbose=True, factor=0.75
        )
    elif model.name in [efficientnet.name]:
        lr = 0.1
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4
        )

        eta_min = 0.0001
        mlflow.log_param("eta_min", eta_min)

        def get_scheduler(optimizer, n_iter_per_epoch):
            cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                eta_min=eta_min,  #  0.000001,
                T_max=(num_epochs - 0 - 20) * n_iter_per_epoch,
            )
            scheduler = GradualWarmupScheduler(
                optimizer,
                multiplier=16,
                total_epoch=20 * n_iter_per_epoch,
                after_scheduler=cosine_scheduler,
            )
            return scheduler

        scheduler = get_scheduler(optimizer, len(train_loader))

    else:
        logger.warn(f"Unknown model {model.name}... Using default optimizer")
        lr = 0.001
        patience = 20
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=patience, verbose=True, factor=0.5
        )

    # optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_history = []
    t = time()

    for epoch in range(num_epochs):
        logger.info(
            f"Starting epoch {epoch} ({time()-t} secs) and lr = {[param['lr'] for param in optimizer.param_groups]}"
        )
        t = time()
        model.set_train_mode()
        mlflow.log_metric(
            "lr", [param["lr"] for param in optimizer.param_groups][0], step=epoch
        )

        if custom_scheduler_step is not None:
            custom_scheduler_step(optimizer, epoch)

        if (epoch == 0) and (prune_percentile > 0.0):
            # mask_ = None
            model, mask_ = prune_model(
                model,
                percentile=prune_percentile,
                init_weight=init_weight_dict,
                zero_grad=False,
            )
        elif (epoch == 0) and (prune_percentile == 0.0):
            mask_ = None

        train_loss = list()
        for i_batch, (x_batch, y_batch) in enumerate(train_loader):
            # Training !!
            # if epoch == 0:
            # mask_ = None
            loss = go_training(
                model,
                x_batch,
                y_batch,
                epoch,
                optimizer,
                loss_func,
                train_noise=train_noise,
                prune_percentile=prune_percentile,
                first_pruned_iter=first_pruned_iter,
                mask_=mask_,
            )
            train_loss.append(loss)
        mlflow.log_metric("train_loss", np.mean(train_loss), step=epoch)

        model.set_eval_mode()
        val_losses = list()
        num_val_batches = 0
        for x_val, y_val in val_loader:
            y_val = y_val.to(device)
            x_val = x_val.type(default_tensor_type)
            y_val_pred = model(x_val)
            val_loss = loss_func(y_val_pred, y_val)
            val_losses.append(val_loss.item())
            num_val_batches += 1
            #  if num_val_batches > 10:
            #      break
        val_loss = np.mean(val_losses)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        logger.info(f"Validation loss = {np.around(val_loss, decimals=4)}")
        loss_history.append(val_loss)

        if (prune_percentile == 0.0) or (epoch > first_pruned_iter * nb_iter_prune):
            step = epoch * len(train_loader) + i_batch
            # step = (epoch % 300) * len(train_loader) + i_batch
            if scheduler is not None:
                scheduler.step(step)
        if epoch % 10 == 9:
            acc = compute_val_acc(model, val_loader)
            logger.info(f"Val acc = {acc}")
            mlflow.log_metric("val_acc", acc, step=epoch)
        if epoch > 0:
            save_pruned_model(
                model,
                current_pruned_percentile,
                first_pruned_iter,
                tot_prune_percentile,
                epoch,
                nepochs,
            )
        if (
            (epoch + 1) % first_pruned_iter == 0
            and epoch != 0
            and prune_percentile > 0.0
            and epoch < first_pruned_iter * nb_iter_prune
        ):
            logger.info(f"Pruned net epoch {epoch}")
            model, mask_ = prune_model(
                model, percentile=prune_percentile, init_weight=init_weight_dict
            )
        # c = 0
        # for i, p in enumerate(model.parameters()):
        #    c += np.count_nonzero(p.data.cpu())
        # current_pruned_percentile = c / sum(p.numel() for p in model.parameters())
        # logger.info(f"percentage non zero parameters = {current_pruned_percentile}")
        # assert model.tot_prune_percentile == np.round(1.0 - current_pruned_percentile, 2)
        count_tot_ = 0
        count_nonzero_ = 0
        for (name, param) in model.named_parameters():
            if len(param.data.size()) > 1:
                count_tot_ += np.prod(param.data.size())
                count_nonzero_ += np.count_nonzero(param.data.cpu())
        pruned_count_ = np.round(1 - count_nonzero_ / count_tot_, 2)
        current_pruned_percentile = 1 - pruned_count_
        logger.info(
            f"Percentgage of zero parameters = {pruned_count_} and model pruned param = {model.tot_prune_percentile}"
        )

    # Saving model
    torch.save(model, model.get_model_savepath())
    mlflow.log_artifact(model.get_model_savepath(), "models")

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
    layers_to_consider: Optional[List[int]] = list(),
) -> Architecture:

    loss_func = nn.CrossEntropyLoss().type(default_tensor_type)

    if dataset.name == "CIFAR100":
        architecture.set_train_mode()
        optimizer = optim.SGD(architecture.parameters(), lr=0.001)
        for i_batch, (x_batch, y_batch) in enumerate(dataset.train_loader):
            x_batch = x_batch.type(default_tensor_type)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            y_pred = architecture(x_batch)
            loss = loss_func(y_pred, y_batch)
            loss.backward()
            optimizer.step()
        architecture.set_eval_mode()
        architecture.is_trained = True

        # Build matrices
        x = dataset.train_dataset[0][0].to(device)
        architecture.forward(x, store_for_graph=False, output="final")
        if layers_to_consider is not None:
            architecture.set_layers_to_consider(layers_to_consider)
        assert architecture.matrices_are_built is True
        return architecture

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
        architecture.train_noise = 0.0
        architecture.tot_prune_percentile = 0.0

        return architecture, loss_func

    architecture.epochs = num_epochs
    architecture.train_noise = train_noise
    architecture.tot_prune_percentile = 0  # tot_prune_percentile

    if not os.path.exists(f"{rootpath}/trained_models"):
        os.mkdir(f"{rootpath}/trained_models")

    try:
        if force_retrain:
            raise FileNotFoundError("Force retrain")
        if dataset.name == "cifar100":
            logger.info(f"Loaded pretrained model for CIFAR100")
            architecture.set_train_mode()
            optimizer = optim.SGD(architecture.parameters(), lr=0.001)
            for i_batch, (x_batch, y_batch) in enumerate(dataset.train_loader):
                x_batch = x_batch.type(default_tensor_type)
                y_batch = y_batch.to(device)
                optimizer.zero_grad()
                y_pred = architecture(x_batch)
                loss = loss_func(y_pred, y_batch)
                loss.backward()
                optimizer.step()
            architecture.set_eval_mode()
        else:
            architecture = torch.load(
                architecture.get_model_savepath(), map_location=device
            )
            logger.info(
                f"Loaded successfully model from {architecture.get_model_savepath()}"
            )
        logger.info(
            f"Test accuracy = {compute_test_acc(architecture, dataset.test_loader)}"
        )
    except FileNotFoundError:
        logger.info(
            f"Unable to find model in {architecture.get_model_savepath()}... Retraining it..."
        )

        # Load already pre-trained efficient net model
        # pretrained_path = "/mnt/nfs/home/m.goibert/TDA_for_adv_robustness/trained_models/efficientnet_e_100.model"
        # architecture = torch.load(pretrained_path, map_location=device)
        # assert architecture.matrices_are_built is True
        # logger.info(f"Correctly downloaded pre-trained model")

        # Train the NN
        #  with mlflow.start_run(run_name=f"Train {architecture.name}"):
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

        # Compute accuracies
        val_accuracy = compute_val_acc(architecture, dataset.val_loader)
        logger.info(f"Validation accuracy = {val_accuracy}")
        test_accuracy = compute_test_acc(architecture, dataset.test_loader)
        logger.info(f"Test accuracy = {test_accuracy}")

    # Forcing eval mode just in case it was not done before
    architecture.set_eval_mode()
    logger.info(f"set archi in eval mode")
    architecture.is_trained = True

    # Build matrices
    x = dataset.train_dataset[0][0].to(device)
    architecture.forward(x, store_for_graph=False, output="final")
    if layers_to_consider is not None:
        architecture.set_layers_to_consider(layers_to_consider)
    assert architecture.matrices_are_built is True

    if with_details:
        return architecture, val_accuracy, test_accuracy

    return architecture


def save_pruned_model(
    architecture,
    current_pruned_percentile,
    first_pruned_iter,
    tot_prune_percentile,
    epoch,
    num_epochs,
):
    current_pruned_percentile = np.round(current_pruned_percentile, 2)
    if (
        (tot_prune_percentile > 0.0)
        and (epoch > 0)
        and ((epoch + 1) % (first_pruned_iter) == 0)
    ):
        logger.info(
            f"Save intermediate pruned model at {architecture.get_model_savepath()}"
        )
        torch.save(architecture, architecture.get_model_savepath())
        logger.info(f"Model correctly saved")


def prune_model(model, percentile=0.1, init_weight=None, zero_grad=True):
    percentile = 100 * percentile
    count_nonzero = 0
    count_tot = 0
    mask_dict = dict()
    num_param = len(list(model.parameters()))
    for i, (name, param) in enumerate(model.named_parameters()):
        # We only prune weight parameters (not bias)
        if len(param.data.size()) > 1:
            if i < num_param - 1:
                perc = np.percentile(
                    abs(param.data.cpu()[param.data.cpu() != 0.0]), percentile
                )
            else:
                perc = np.percentile(
                    abs(param.data.cpu()[param.data.cpu() != 0.0]), percentile / 2.0
                )
            mask = (
                torch.tensor(np.where(abs(param.data.cpu()) < perc, 0, 1))
                .type(default_tensor_type)
                .to(device)
            )
            mask_dict[i] = mask
            param.data = mask * param.data
            if zero_grad:
                param.grad.data = mask * param.grad.data
            if init_weight:  # In case of reinitialization of weights
                param.data = mask * init_weight[str(name)]
            count_tot += np.prod(param.data.size())
            count_nonzero += np.count_nonzero(param.data.cpu())
    logger.info(f"Percentage pruned = {1 - count_nonzero/count_tot}")
    pruned_count = np.round(1 - count_nonzero / count_tot, 2)
    model.tot_prune_percentile = pruned_count
    logger.info(f"architecture pruned percentile = {model.tot_prune_percentile}")

    return model, mask_dict
