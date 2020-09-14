from tda.protocol import get_protocolar_datasets
from tda.dataset.datasets import Dataset
from tda.models.architectures import get_architecture, mnist_mlp
from tda.models import get_deep_model


def test_protocolar_datasets(
    dataset_size=10, epsilons=[0.01, 0.05], attack_type="FGSM", noise=0.2, succ_adv=True
):
    dataset = Dataset("MNIST")
    model = get_deep_model(
        dataset=dataset, architecture=get_architecture(mnist_mlp.name), num_epochs=25
    )
    train_clean, test_clean, train_adv, test_adv = get_protocolar_datasets(
        noise=noise,
        dataset=dataset,
        succ_adv=succ_adv,
        archi=model,
        dataset_size=dataset_size,
        attack_type=attack_type,
        all_epsilons=epsilons,
    )

    print(len(train_clean))
    print(len(test_clean))
    for key in train_adv:
        print(f"{key} => {len(train_adv[key])}")
    for key in test_adv:
        print(f"{key} => {len(test_adv[key])}")
