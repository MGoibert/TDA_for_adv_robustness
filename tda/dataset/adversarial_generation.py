from tda.tda_logging import get_logger

import torch
from tda.devices import device
from tda.models import Architecture, Dataset

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

def guide_image(y, dataset_name="MNIST", num_classes=10):
    dataset = Dataset(dataset_name.upper())
    guide_dict = dict()
    for i in range(0,10):
        guide_dict[i] = 0.0
    state = 0
    i = 0
    while state < 10:
        im, cl = dataset.test_and_val_dataset[i]
        if isinstance(guide_dict[cl], float):
            guide_dict[cl] = im 
            state += 1
        i+=1
    
    logger.info(f"guide -- y = {y} and {isinstance(y,int)}")
    if isinstance(y, int):
        target_class = (y+1) % num_classes
        guide = guide_dict[target_class]
    else:
        target_class = [(y_+1) % num_classes for y_ in y]
        guide = [guide_dict[target_class_] for target_class_ in target_class]
        guide = torch.stack(guide)
    return guide, target_class

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
    BOUNDARY = "BOUNDARY"
    FEATUREADVERSARIES = "FEATUREADVERSARIES"

    @staticmethod
    def require_epsilon(attack_type: str) -> bool:
        return attack_type in [AttackType.FGSM, AttackType.PGD]


def adversarial_generation(
    model: Architecture,
    x,
    y,
    epsilon=0.25,
    attack_type=AttackType.FGSM,
    num_iter=10,
    attack_backend: str = AttackBackend.FOOLBOX,
    dataset_name: str = "MNIST",
):
    """
    Create an adversarial example (FGMS only for now)
    """
    x.requires_grad = True

    logger.info(f"Generating for x (shape={x.shape}) and y (shape={y.shape})")

    if attack_backend == AttackBackend.ART:

        from art.attacks.evasion import (
            FastGradientMethod,
            ProjectedGradientDescent,
            DeepFool as DeepFoolArt,
            CarliniL2Method,
            HopSkipJump,
            FeatureAdversariesPyTorch,
        )
        #from art.attacks.evasion.feature_adversaries import FeatureAdversariesPytorch
        #from adversarial-robustness-toolbox.art.attacks.evasion.feature_adversaries import feature_adversaries_pytorch

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
        elif attack_type == AttackType.FEATUREADVERSARIES:
            logger.info(f"IN FEATURE ADV : type model = {type(model.art_classifier)}")
            attacker = FeatureAdversariesPyTorch(
                estimator=model.art_classifier,
                delta=float(epsilon),
                optimizer=None,
                layer=0,
                max_iter=num_iter,
                step_size= 2*epsilon/num_iter,
                batch_size=32,
            )
        else:
            raise NotImplementedError(f"{attack_type} is not available in ART")
        
        if attack_type == AttackType.FEATUREADVERSARIES:
            guide, target_class = guide_image(y=y, dataset_name=dataset_name.upper(), num_classes=10)
            logger.info(f"y = {y} and guide class = {target_class}")
            logger.info(f"x shape = {x.shape} and guide shape = {guide.shape}")
            attacked = attacker.generate(x=x.detach().cpu().numpy(), y=guide.detach().cpu().numpy())
        else:
            attacked = attacker.generate(x=x.detach().cpu().numpy())
        attacked = torch.from_numpy(attacked).to(device)

    elif attack_backend == AttackBackend.FOOLBOX:

        import foolbox as fb

        if model.name in ["efficientnet", "resnet32", "resnet44", "resnet56"]:
            model.set_default_forward_mode(None)
        else:
            model.set_default_forward_mode("presoft")

        if attack_type == AttackType.FGSM:
            attacker = fb.attacks.LinfFastGradientAttack()
        elif attack_type == AttackType.PGD:
            attacker = fb.attacks.LinfProjectedGradientDescentAttack(
                steps=50, random_start=False, rel_stepsize=2 / 50
            )
        elif attack_type == AttackType.DeepFool:
            attacker = fb.attacks.LinfDeepFoolAttack(loss="crossentropy")
        elif attack_type == AttackType.CW:
            attacker = fb.attacks.L2CarliniWagnerAttack(steps=num_iter)
        elif attack_type == AttackType.BOUNDARY:
            attacker = fb.attacks.BoundaryAttack(
                steps=7000, spherical_step=0.01, source_step=0.01
            )
            x = x.float()
        else:
            raise NotImplementedError(f"{attack_type} is not available in Foolbox")

        attacked, _, _ = attacker(
            model.foolbox_classifier,
            x.detach(),
            torch.from_numpy(y).to(device),
            epsilons=epsilon,
        )
        #samenb = 0
        #for x_, att_ in zip(x, attacked):
        #    if (x_==att_).all():
        #        samenb += 1
        #logger.info(f"attacker = {attacker} with epsilon = {epsilon} and nb sames = {samenb}")

        model.set_default_forward_mode(None)

    elif attack_backend == AttackBackend.CUSTOM:

        from tda.dataset.custom_attacks import FGSM, BIM, DeepFool, CW

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

    return attacked.detach().double()
