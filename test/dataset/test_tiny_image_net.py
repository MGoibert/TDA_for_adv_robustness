from tda.dataset.tiny_image_net import load_tiny_image_net
from tda.dataset.datasets import Dataset


def test_load_tiny_image_net():
    train_dataset = load_tiny_image_net(mode="test")
    assert len(train_dataset) == 10000


def test_datasets():

    my_source_dataset = Dataset("TinyImageNet")

    for sample, labels in my_source_dataset.train_loader:
        print(sample.shape)
        print(labels.shape)
        assert sample.shape == (128, 3, 64, 64)
        break
