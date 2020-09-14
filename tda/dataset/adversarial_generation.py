from art.attacks.evasion import (
    FastGradientMethod,
    ProjectedGradientDescent,
    DeepFool as DeepFoolArt,
    CarliniL2Method,
    HopSkipJump,
)

from tda.dataset.custom_attacks import FGSM, BIM, DeepFool, CW
from tda.tda_logging import get_logger

import foolbox as fb
import torch
from tda.devices import device
from tda.models import Architecture

logger = get_logger("AdvGen")

# One-hot vector based on scalar
def one_hot(y, num_classes=None):
    if num_classes is None:
        classes, _ = y.max(0)
        num_classes = (classes.max() + 1).item()
    if y.dim() > 0:
        y_ = torch.zeros(len(y), num_classes, device=y.device)
    else:
        y_ = torch.zeros(1, num_classes)
    y_.scatter_(1, y.unsqueeze(-1), 1)
    y_ = y_.to(device)
    return y_


def ce_loss(outputs, labels, num_classes=None):
    """
    Cross_entropy loss
    (output = post-softmax output of the model,
     and label =  one-hot)
    """
    labels = one_hot(labels, num_classes=num_classes)
    size = len(outputs)
    if outputs[0].dim() == 0:
        for i in range(size):
            outputs[i] = outputs[i].unsqueeze(-1)
    if labels[0].dim() == 0:
        for i in range(size):
            labels[i] = labels[i].unsqueeze(-1)

    res = (
        1.0
        / size
        * sum([torch.dot(torch.log(outputs[i]), labels[i]) for i in range(size)])
    )
    return -res


class AttackBackend(object):
    CUSTOM = "CUSTOM"
    FOOLBOX = "FOOLBOX"
    ART = "ART"


class AttackType(object):
    FGSM = "FGSM"
    PGD = "PGD"
    DeepFool = "DeepFool"
    CW = "CW"
    SQUARE = "SQUARE"
    HOPSKIPJUMP = "HOPSKIPJUMP"

    @staticmethod
    def require_epsilon(attack_type: "AttackType") -> bool:
        return attack_type in [AttackType.FGSM, AttackType.PGD]


def adversarial_generation(
    model: Architecture,
    x,
    y,
    epsilon=0.25,
    attack_type=AttackType.FGSM,
    num_iter=10,
    attack_backend: str = AttackBackend.ART,
):
    """
    Create an adversarial example (FGMS only for now)
    """
    x.requires_grad = True

    logger.info(f"Generating for x (shape={x.shape}) and y (shape={y.shape})")

    if attack_backend == AttackBackend.ART:
        if attack_type == AttackType.FGSM:
            attacker = FastGradientMethod(estimator=model.art_classifier, eps=epsilon)
        elif attack_type == AttackType.PGD:
            attacker = ProjectedGradientDescent(
                estimator=model.art_classifier,
                max_iter=num_iter,
                eps=epsilon,
                eps_step=2 * epsilon / num_iter,
            )
        elif attack_type == AttackType.DeepFool:
            attacker = DeepFoolArt(classifier=model.art_classifier, max_iter=num_iter)
        elif attack_type == "CW":
            attacker = CarliniL2Method(
                classifier=model.art_classifier,
                max_iter=num_iter,
                binary_search_steps=15,
            )
        elif attack_type == AttackType.SQUARE:
            # attacker = SquareAttack(estimator=model.get_art_classifier())
            raise NotImplementedError("Work in progress")
        elif attack_type == AttackType.HOPSKIPJUMP:
            attacker = HopSkipJump(
                classifier=model.art_classifier,
                targeted=False,
                max_eval=100,
                max_iter=10,
                init_eval=10,
            )
        else:
            raise NotImplementedError(f"{attack_type} is not available in ART")

        attacked = attacker.generate(x=x.detach().cpu())
        attacked = torch.from_numpy(attacked).to(device)

    elif attack_backend == AttackBackend.FOOLBOX:
        if attack_type == AttackType.FGSM:
            attacker = fb.attacks.LinfFastGradientAttack()
        elif attack_type == AttackType.PGD:
            attacker = fb.attacks.LinfProjectedGradientDescentAttack()
        elif attack_type == AttackType.DeepFool:
            attacker = fb.attacks.LinfDeepFoolAttack()
        elif attack_type == AttackType.CW:
            attacker = fb.attacks.L2CarliniWagnerAttack(steps=num_iter)
        else:
            raise NotImplementedError(f"{attack_type} is not available in Foolbox")

        attacked, _, _ = attacker(
            model.foolbox_classifier,
            x.detach(),
            torch.from_numpy(y).to(device),
            epsilons=epsilon,
        )

    elif attack_backend == AttackBackend.CUSTOM:
        if attack_type == AttackType.FGSM:
            attacker = FGSM(model, ce_loss)
            attacked = attacker.run(
                data=x.detach(), target=torch.from_numpy(y).to(device), epsilon=epsilon
            )
        elif attack_type == AttackType.PGD:
            attacker = BIM(model, ce_loss, lims=(0, 1), num_iter=num_iter)
            attacked = attacker.run(
                data=x.detach(), target=torch.from_numpy(y).to(device), epsilon=epsilon
            )
        elif attack_type == AttackType.DeepFool:
            attacker = DeepFool(model, num_classes=10, num_iter=num_iter)
            attacked = [
                attacker(x[i].detach(), torch.tensor(y[i]).to(device))
                for i in range(len(x))
            ]
            attacked = torch.cat([torch.unsqueeze(a, 0) for a in attacked], 0)
        elif attack_type == AttackType.CW:
            attacker = CW(model, lims=(0, 1), num_iter=num_iter)
            attacked = attacker.run(
                data=x.detach(), target=torch.from_numpy(y).to(device)
            )
            attacked = torch.cat([torch.unsqueeze(a, 0) for a in attacked], 0)
        else:
            raise NotImplementedError(
                f"{attack_type} is not available as custom implementation"
            )
    else:
        raise NotImplementedError(f"Unknown backend {attack_backend}")

    # x_adv = torch.cat([_to_tensor(x) for x in attacked], 0).to(
    #    device
    # )

    return attacked.detach()
