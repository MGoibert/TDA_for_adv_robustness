from tda.graph_dataset import get_sample_dataset
from tda.models.datasets import Dataset
from tda.models import get_deep_model
from tda.models.architectures import mnist_mlp, get_architecture

source_dataset = Dataset("MNIST")


def test_get_sample_dataset():

    dataset = get_sample_dataset(
        epsilon=0.1,
        noise=0.0,
        adv=True,
        dataset=source_dataset,
        archi=get_deep_model(
            num_epochs=25,
            dataset=source_dataset,
            architecture=get_architecture(mnist_mlp.name)),
        dataset_size=5,
        train=False,
        succ_adv=True,
        attack_type="FGSM",
        num_iter=10,
        offset=0,
        per_class=False
    )

    assert len(dataset) == 5

    for line in dataset:
        assert line.y != line.y_pred

    print(dataset)
