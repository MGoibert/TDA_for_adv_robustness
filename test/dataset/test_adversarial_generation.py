from tda.dataset.adversarial_generation import (
    adversarial_generation,
    AttackBackend,
    AttackType,
)
from tda.models.architectures import cifar_lenet
import torch
import pytest
import numpy as np


@pytest.mark.parametrize(
    "attack_type,attack_backend",
    [
        # CUSTOM
        (AttackType.FGSM, AttackBackend.CUSTOM),
        (AttackType.DeepFool, AttackBackend.CUSTOM),
        (AttackType.PGD, AttackBackend.CUSTOM),
        (AttackType.CW, AttackBackend.CUSTOM),
        # ART
        (AttackType.FGSM, AttackBackend.ART),
        (AttackType.DeepFool, AttackBackend.ART),
        (AttackType.PGD, AttackBackend.ART),
        (AttackType.CW, AttackBackend.ART),
        (AttackType.HOPSKIPJUMP, AttackBackend.ART),
        # FOOLBOX
        (AttackType.FGSM, AttackBackend.FOOLBOX),
        (AttackType.DeepFool, AttackBackend.FOOLBOX),
        (AttackType.PGD, AttackBackend.FOOLBOX),
        (AttackType.CW, AttackBackend.FOOLBOX),
    ],
)
def test_generate(attack_type, attack_backend):

    x = torch.ones((32, 3, 32, 32))
    y = np.array([1 for _ in range(32)])

    adversarial_generation(
        x=x,
        y=y,
        epsilon=0.25,
        attack_type=attack_type,
        num_iter=10,
        attack_backend=attack_backend,
        model=cifar_lenet,
    )
