from tda.models import get_deep_model
from tda.models.datasets import Dataset


def test_get_mnist_model():
    source_dataset = Dataset("MNIST")
    get_deep_model(dataset=source_dataset, num_epochs=1)

