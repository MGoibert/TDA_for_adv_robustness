from tda.protocol import get_protocolar_datasets
from tda.models.datasets import Dataset
from tda.models.architectures import get_architecture, mnist_mlp
from tda.models import get_deep_model


def test_protocolar_datasets():
    dataset = Dataset("MNIST")

    train_clean, test_clean, train_adv, test_adv = get_protocolar_datasets(
        noise=0.2,
        dataset=dataset,
        succ_adv=True,
        archi=get_deep_model(
            dataset=dataset,
            architecture=get_architecture(mnist_mlp.name),
            num_epochs=25),
        dataset_size=10,
        attack_type="FGSM",
        all_epsilons=[0.01, 0.05]
    )

    print(len(train_clean))
    print(len(test_clean))
    for key in train_adv:
        print(f"{key} => {len(train_adv[key])}")
    for key in test_adv:
        print(f"{key} => {len(test_adv[key])}")
