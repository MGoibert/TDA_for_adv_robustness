from torch import nn, optim, no_grad
import torch
import numpy as np
from tqdm import tqdm
import pathlib
import os
import logging
from tda.models.architectures import mnist_mlp, Architecture
from tda.models.datasets import Dataset
from tda.rootpath import rootpath
logger = logging.getLogger()

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
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    acc = correct / len(test_loader.dataset)
    print("Test accuracy =", acc)
    return acc


def train_network(model, train_loader, val_loader, loss_func, num_epochs, train_noise=0.0):
    """
    Helper function to train an arbitrary model
    """
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss_history = []
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, verbose=True,
        factor=0.5)
    for epoch in range(num_epochs):
        model.train()
        train_loader = tqdm(train_loader)
        for x_batch, y_batch in train_loader:
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


def get_deep_model(
        num_epochs: int,
        dataset: Dataset,
        architecture: Architecture = mnist_mlp,
        train_noise: float = 0.0
) -> (nn.Module, nn.Module):
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
    loss_func = nn.CrossEntropyLoss()

    try:
        net = torch.load(model_filename)
        print(f"Loaded successfully model from {model_filename}")
    except FileNotFoundError:
        print(f"Unable to find model in {model_filename}... Retraining it...")

        # Train the NN
        net = train_network(
            architecture,
            dataset.train_loader,
            dataset.val_loader,
            loss_func,
            num_epochs,
            train_noise)[0]

        # Compute accuracies
        logger.info(f"Validation accuracy = {compute_val_acc(architecture, dataset.val_loader)}")
        logger.info(f"Test accuracy = {compute_test_acc(architecture, dataset.test_loader)}")

        # Saving model
        torch.save(net, model_filename)

    return net, loss_func
