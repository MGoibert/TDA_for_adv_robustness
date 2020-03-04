import torch
import torch.nn as nn
import numpy as np
import random
from collections import OrderedDict

from tda.models import get_deep_model
from tda.models.datasets import Dataset
from tda.models.architectures import * #mnist_mlp, svhn_cnn_simple, Architecture, mnist_lenet
from tda.threshold_underoptimized_edges import process_thresholds_underopt


def test_get_mnist_model():
    torch.manual_seed(37)
    random.seed(38)
    np.random.seed(39)

    source_dataset = Dataset("MNIST")
    _, val_acc, test_acc = get_deep_model(
        dataset=source_dataset.Dataset_,
        num_epochs=1,
        architecture=mnist_lenet,
        with_details=True,
        force_retrain=True
    )
    print(val_acc)
    print(test_acc)


def test_get_svhn_model():
    source_dataset = Dataset("SVHN")
    get_deep_model(
        dataset=source_dataset.Dataset_,
        num_epochs=2,
        architecture=svhn_cnn_simple
    )


def test_train_eval():
    archi: Architecture = mnist_mlp
    archi.set_train_mode()
    training_modes = [layer.func.training for layer in archi.layers]
    print(training_modes)
    assert all(training_modes)
    archi.set_eval_mode()
    eval_modes = [not layer.func.training for layer in archi.layers]
    print(eval_modes)
    assert all(eval_modes)

def resave_model(file):
    class the_lenet(nn.Module):
        def __init__(self):
            super(the_lenet, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, kernel_size=5, bias=False)
            self.conv2 = nn.Conv2d(6, 16, kernel_size=5, bias=False)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = x.double()
            x = F.max_pool2d(F.relu(self.conv1(x), 2))
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            x = F.softmax(x, dim=1)
            return x

    my_mlp = the_lenet()
    #model = torch.load(file, map_location=torch.device('cpu'))
    model = svhn_lenet
    model.load_state_dict(torch.load(file, map_location=torch.device('cpu')))
    print(f"Loaded models")

    print(f"model state dict = {model.state_dict().keys()}")
    #print(f"model = {model.keys()}")
    print(f"my model state dict = {my_mlp.state_dict().keys()}")
    nmodel = OrderedDict()
    nmodel['conv1.weight'] = model.state_dict()['layer0_weight']
    nmodel['conv2.weight'] = model.state_dict()['layer2_weight']
    nmodel['fc1.weight'] = model.state_dict()['layer4_weight']
    nmodel['fc1.bias'] = model.state_dict()['layer4_bias']
    nmodel['fc2.weight'] = model.state_dict()['layer5_weight']
    nmodel['fc2.bias'] = model.state_dict()['layer5_bias']
    nmodel['fc3.weight'] = model.state_dict()['layer6_weight']
    nmodel['fc3.bias'] = model.state_dict()['layer6_bias']
    #nmodel['fc3.weight'] = model.state_dict()['layer2_weight']
    #nmodel['fc3.bias'] = model.state_dict()['layer2_bias']
    my_mlp.load_state_dict(nmodel)
    torch.save(my_mlp.state_dict(), "/Users/m.goibert/Documents/temp/gram_mat/init_svhn_lenet_model_205_epochs.model")
    print(f"Done !!")

def test_conv():
    cnn = Architecture(
    name="cnn",
    layers=[
        ConvLayer(1, 2, 2, bias=False, name="conv1"),  # output 6 * 28 * 28
        ConvLayer(2, 3, 2, bias=False, name="conv2"),  # output 6 * 28 * 28
        LinearLayer(3, 2, name="fc1"),
        SoftMaxLayer()
    ])
    model = cnn
    for p in model.parameters():
        logger.info(f"p = {p}")
    x = torch.round(10*torch.randn([1,1,3,3]))
    logger.info(f"x = {x}")
    out = cnn(x)
    logger.info(f"out = {out}")
    logger.info(f"{np.round(cnn.get_graph_values(x)[(0,1)].todense(),2)}")

def test_new_threshold():
    architecture = get_architecture(mnist_lenet.name)
    dataset = Dataset.get_or_create(name="MNIST")

    architecture = get_deep_model(
        num_epochs=53,
        dataset=dataset,
        architecture=architecture,
        train_noise=0.0
    )

    thresholds = process_thresholds_underopt(
        raw_thresholds="0.1_0.1_0.1_0.1_0.1",
        architecture=architecture,
    )


if __name__ == "__main__":
    #resave_model("/Users/m.goibert/Documents/Criteo/P2_TDA_Detection/TDA_for_adv_robustness/trained_models/init_svhn_svhn_lenet_205_epochs.model")
    test_new_threshold()



