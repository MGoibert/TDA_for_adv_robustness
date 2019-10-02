from tda.models import get_deep_model
from tda.models.datasets import Dataset
from tda.models.architectures import mnist_mlp, svhn_mlp


def test_get_mnist_model():
    source_dataset = Dataset("MNIST")
    get_deep_model(
        dataset=source_dataset,
        num_epochs=1,
        architecture=mnist_mlp
    )


def test_get_svhn_model():
    source_dataset = Dataset("SVHN")
    get_deep_model(
        dataset=source_dataset,
        num_epochs=1,
        architecture=svhn_mlp
    )

