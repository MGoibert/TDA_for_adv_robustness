import typing

from sklearn.model_selection import train_test_split

from tda.graph_dataset import get_sample_dataset
from tda.models import Architecture, Dataset


def get_protocolar_datasets(
        noise: float,
        dataset: Dataset,
        succ_adv: bool,
        archi: Architecture,
        dataset_size: int,
        attack_type: str,
        all_epsilons: typing.List
):
    train_clean = get_sample_dataset(
        adv=False,
        epsilon=0.0,
        noise=0.0,
        dataset=dataset,
        train=False,
        succ_adv=succ_adv,
        archi=archi,
        dataset_size=dataset_size // 2,
        offset=0
    )

    if False: #noise > 0.0:
        train_clean += get_sample_dataset(
            adv=False,
            epsilon=0.0,
            noise=noise,
            dataset=dataset,
            train=False,
            succ_adv=succ_adv,
            archi=archi,
            dataset_size=dataset_size // 2,
            offset=0
        )

    test_clean = get_sample_dataset(
        adv=False,
        epsilon=0.0,
        noise=0.0,
        dataset=dataset,
        train=False,
        succ_adv=succ_adv,
        archi=archi,
        dataset_size=dataset_size // 2,
        offset=dataset_size // 2
    )

    if False: #noise > 0.0:
        test_clean += get_sample_dataset(
            adv=False,
            epsilon=0.0,
            noise=noise,
            dataset=dataset,
            train=False,
            succ_adv=succ_adv,
            archi=archi,
            dataset_size=dataset_size // 2,
            offset=dataset_size // 2
        )

    train_adv = dict()
    test_adv = dict()

    for epsilon in all_epsilons:

        adv = get_sample_dataset(
            adv=True,
            noise=0.0,
            dataset=dataset,
            train=False,
            succ_adv=succ_adv,
            archi=archi,
            attack_type=attack_type,
            epsilon=epsilon,
            num_iter=50,
            dataset_size=dataset_size,
            offset=dataset_size
        )

        train_adv[epsilon], test_adv[epsilon] = train_test_split(adv, test_size=0.5, random_state=37)

    return train_clean, test_clean, train_adv, test_adv
