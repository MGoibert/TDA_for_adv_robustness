import os

import torch
import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self, num_classes=10, name="LeNet"):
        super(LeNet, self).__init__()
        self.num_classes = num_classes
        self.name = name
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(500, num_classes)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.maxpool1(x)
        x = self.relu2(self.conv2(x))
        x = self.maxpool2(x)
        x = x.view(-1, 4 * 4 * 50)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

    @property
    def save_path(self):
        return "%s_pretrained.pth" % self.name

    def save(self, path=None):
        if path is None:
            path = self.save_path
        torch.save(self.state_dict(), path)
        print("Saved %s" % path)

    def train_or_load(self, path=None, retrain=False, train_loader=None,
                      val_data=None, num_epochs=50, save=True,
                      **kwargs):
        if path is None:
            path = self.save_path
        if not retrain and os.path.exists(path):
            self.load_state_dict(torch.load(path))
            print("Loaded %s" % path)
        else:
            import sys
            sys.path.append("../safe_ml/code")
            import utils

            if not retrain:
                print("Couldn't find model file %s; retraining..." % (
                    path))
            if train_loader is None:
                raise ValueError("Need to provide a value for train_loader!")
            utils.train_net(self, train_loader, val_data=val_data,
                            num_epochs=num_epochs)
            if save:
                self.save(path)
