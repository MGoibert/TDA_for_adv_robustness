from torch import nn, optim, no_grad
import torch
import numpy as np
from time import time
import pathlib
import os
import copy
import typing
from tda.models.architectures import mnist_mlp, Architecture, mnist_lenet, svhn_lenet, cifar_lenet, fashion_mnist_lenet
from tda.models.datasets import Dataset
from tda.rootpath import rootpath
from tda.devices import device
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
            if device.type == "cuda":
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
            if device.type == "cuda":
                data = data.to(device)
                target = target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    acc = correct / len(test_loader.dataset)
    print("Test accuracy =", acc)
    return acc


def train_network(
        model: Architecture,
        train_loader,
        val_loader,
        loss_func,
        num_epochs: int,
        train_noise: float = 0.0,
        prune_percentile: float = 0.0,
        first_pruned_iter: int = 1) -> Architecture:
    """
    Helper function to train an arbitrary model
    """
    # Save model initial values
    if True:
        filename = f"{rootpath}/trained_models/init_svhn_" \
                     f"svhn_lenet_" \
                     f"103_" \
                     f"epochs.model"
        torch.save(model.state_dict(), filename)
        logger.info(f"Saved initial model in {filename}")

    if device.type == "cuda":
        logger.info(f"Learning on GPU {device}")
        model.cuda(device)
    else:
        logger.info("Learning on CPU")

    if prune_percentile != 0.0:
        init_weight_dict = copy.deepcopy(model.state_dict())

    if model.name in [mnist_lenet.name, fashion_mnist_lenet.name]:
        lr = 0.1
        patience = 12
    elif model.name == svhn_lenet.name:
        lr = 0.05
        patience = 8
    elif model.name == mnist_mlp.name:
        lr = 0.1
        patience = 5
    elif model.name == cifar_lenet.name:
        lr = 0.2
        patience = 15

    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_history = []
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=patience, verbose=True,
        factor=0.5)
    t = time()

    for epoch in range(num_epochs):
        logger.info(f"Starting epoch {epoch} ({time()-t} secs)")
        t = time()
        model.set_train_mode()

        for x_batch, y_batch in train_loader:
            if device.type == "cuda":
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
            x_batch = x_batch.double()
            if train_noise > 0.0:
                x_batch_noisy = torch.clamp(x_batch + train_noise * torch.randn(x_batch.size()), -0.5, 0.5).double()
                #x_batch = torch.cat((x_batch, x_batch_noisy), 0)
                y_batch_noisy = y_batch
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = loss_func(y_pred, y_batch)
            if train_noise > 0.0:
                y_pred_noisy = model(x_batch_noisy)
                loss_noisy = loss_func(y_pred_noisy, y_batch_noisy)
                loss = 0.75*loss + 0.25*loss_noisy
            loss.backward()

            # c = 0
            for i, (name, param) in enumerate(model.named_parameters()):
                if len(param.data.size()) > 1 and epoch > first_pruned_iter and prune_percentile != 0.0:
                    param.data = param.data * mask_[i]
                    param.grad.data = param.grad.data * mask_[i]

                # c += np.count_nonzero(param.grad.data.cpu())
            # logger.info(f"epoch {epoch} nonzero grad = {c}")
            
            optimizer.step()
        model.set_eval_mode()
        for x_val, y_val in val_loader:
            x_val = x_val.double()
            if device.type == "cuda":
                x_val = x_val.to(device)
                y_val = y_val.to(device)
            y_val_pred = model(x_val)
            val_loss = loss_func(y_val_pred, y_val)
            logger.info(f"Validation loss = {np.around(val_loss.item(), decimals=4)}")
            loss_history.append(val_loss.item())
        if True:#epoch > num_epochs-first_pruned_iter and prune_percentile != 0.0:
            scheduler.step(val_loss)

        if epoch % first_pruned_iter == 0 \
                and epoch != 0 \
                and prune_percentile != 0.0 \
                and epoch < 0.9*(num_epochs-first_pruned_iter):
            logger.info(f"Pruned net epoch {epoch}")
            model, mask_ = prune_model(model,
                                       percentile=prune_percentile,
                                       init_weight=init_weight_dict)
        c = 0
        for i, p in enumerate(model.parameters()):
            c += np.count_nonzero(p.data.cpu())
        logger.info(f"percentage non zero parameters = {c/sum(p.numel() for p in model.parameters())}")

    return model, loss_history


def get_deep_model(
        num_epochs: int,
        dataset: Dataset,
        architecture: Architecture = mnist_mlp,
        train_noise: float = 0.0,
        with_details: bool = False,
        force_retrain: bool = False,
        pretrained_pth: str=None
) -> Architecture:
    loss_func = nn.CrossEntropyLoss()

    if pretrained_pth is not None:
        # Experimental: to help loading an existing model
        # possibly trained outside of our framework
        # Can be used with tda/models/pretrained/lenet_mnist_model.pth
        print(rootpath)
        state_dict = torch.load(
            f"{rootpath}/tda/models/pretrained/{pretrained_pth}",
            map_location=device
        )
        state_dict = {key.replace(".", "_"): state_dict[key] for key in state_dict}
        architecture.load_state_dict(state_dict)
        if device.type == "cuda":
            architecture.cuda(device)
        return architecture, loss_func

    if not os.path.exists(f"{rootpath}/trained_models"):
        os.mkdir(f"{rootpath}/trained_models")

    if train_noise > 0.0:
        nprefix = f"{train_noise}_"
    else:
        nprefix = ""

        model_filename = f"{rootpath}/trained_models/{dataset.name}_" \
                     f"{architecture.name}_" \
                     f"{nprefix}" \
                     f"{num_epochs}_" \
                     f"epochs.model"
    logger.info(f"Filename = {model_filename} \n")

    try:
        if force_retrain:
            raise FileNotFoundError("Force retrain")

        architecture = torch.load(model_filename, map_location=device)
        logger.info(f"Loaded successfully model from {model_filename}")
    except FileNotFoundError:
        logger.info(f"Unable to find model in {model_filename}... Retraining it...")

        # Train the NN
        train_network(
            architecture,
            dataset.train_loader,
            dataset.val_loader,
            loss_func,
            num_epochs,
            train_noise
        )

        # Saving model
        torch.save(architecture, model_filename)

        # Compute accuracies
        val_accuracy = compute_val_acc(architecture, dataset.val_loader)
        logger.info(f"Validation accuracy = {val_accuracy}")
        test_accuracy = compute_test_acc(architecture, dataset.test_loader)
        logger.info(f"Test accuracy = {test_accuracy}")

    if device.type == "cuda":
        architecture.cuda(device)

    if with_details:
        return architecture, val_accuracy, test_accuracy
    # Forcing eval mode just in case it was not done before
    architecture.set_eval_mode()
    architecture.is_trained = True
    return architecture


def prune_model(model, percentile=10, init_weight=None):
    count_nonzero = 0
    count_tot = 0
    mask_dict = dict()
    num_param = len(list(model.parameters()))
    for i, (name, param) in enumerate(model.named_parameters()):
        # We only prune weight parameters (not bias)
        if len(param.data.size()) > 1:
            if i < num_param - 1:
                perc = np.percentile(abs(param.data[param.data != 0.0]), percentile)
            else:
                perc = np.percentile(abs(param.data[param.data != 0.0]), percentile/2.0)
            mask = torch.tensor(np.where(abs(param.data) < perc, 0, 1)).double()
            mask_dict[i] = mask
            param.data = mask * param.data
            param.grad.data = mask * param.grad.data
            if init_weight: # In case of reinitialization of weights
                param.data = mask * init_weight[str(name)]
            count_tot += np.prod(param.data.size())
            count_nonzero += np.count_nonzero(param.data.cpu())
    logger.info(f"Percentage pruned = {1 - count_nonzero/count_tot}")

    return model, mask_dict
