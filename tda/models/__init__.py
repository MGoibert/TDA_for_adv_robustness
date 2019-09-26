from torch import nn, optim, no_grad
import torch
import numpy as np
from tqdm import tqdm
import pathlib
from tda.models.architectures import MNISTMLP
from tda.models.datasets import Dataset

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


def train_network(model, train_loader, val_loader, loss_func, num_epochs):
    """
    Helper function to train an arbitrary model
    """
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss_history = []
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
            print("Validation loss = ", np.around(val_loss.item(), decimals=4))
            loss_history.append(val_loss.item())
        scheduler.step(val_loss)

    return model, loss_history


def get_deep_model(
        num_epochs: int,
        dataset: Dataset
) -> (nn.Module, nn.Module):
    model_filename = f"/tmp/tda/trained_models/mnist_{num_epochs}_epochs.model"
    loss_func = nn.CrossEntropyLoss()

    try:
        net = torch.load(model_filename)
        print(f"Loaded successfully model from {model_filename}")
    except FileNotFoundError:
        print(f"Unable to find model in {model_filename}... Retraining it...")

        # Use the MLP model
        model = MNISTMLP()

        # Train the NN
        net = train_network(
            model,
            dataset.train_loader,
            dataset.val_loader,
            loss_func,
            num_epochs)[0]

        # Compute accuracies
        compute_val_acc(model, dataset.val_loader)
        compute_test_acc(model, dataset.test_loader)

        # Saving model
        torch.save(net, model_filename)

    return net, loss_func
