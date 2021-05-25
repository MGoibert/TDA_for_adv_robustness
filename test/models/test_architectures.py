from tda.models import Architecture
from torchvision.transforms import transforms
import torchvision
import torch
import tempfile
import pandas as pd

from tda.models.architectures import get_architecture


def test_walk_through_dag_list():
    my_edges = [(1, 2), (2, 3), (3, 4)]
    my_order = Architecture.walk_through_dag(my_edges)
    assert my_order == [1, 2, 3, 4]


def test_walk_through_dag_simple():
    my_edges = [
        (1, 2),
        (2, 3),
        (3, 4),
        (6, 5),
        (2, 5),
        (5, 4),  # Shortcut for layer 2 to 4
    ]
    my_order = Architecture.walk_through_dag(my_edges)

    # Check that all the nodes are visited
    for edge in my_edges:
        assert edge[0] in my_order and edge[1] in my_order

    # Check that for all edges the source is visited before the target
    for edge in my_edges:
        assert my_order.index(edge[0]) < my_order.index(edge[1])

    print(Architecture.get_parent_dict(my_edges))


def test_resnets_cifar100_performance():
    _trans = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.507, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2761]
            ),
        ]
    )
    with tempfile.TemporaryDirectory() as tempdir:
        ds_train = torchvision.datasets.CIFAR100(
            train=True, transform=_trans, download=True, root=f"{tempdir}/cifar100"
        )
        ds_test = torchvision.datasets.CIFAR100(
            train=False, transform=_trans, download=True, root=f"{tempdir}/cifar100"
        )
    loaders = {
        "train": torch.utils.data.DataLoader(ds_train, batch_size=256),
        "test": torch.utils.data.DataLoader(ds_test, batch_size=256),
    }

    results = dict()

    for model_name in ["resnet20", "resnet32", "resnet44", "resnet56"]:
        archi = get_architecture(f"cifar100_{model_name}")
        res = dict()
        for mode in loaders:
            corrects = 0
            total = 0
            for images, real_classes in loaders[mode]:
                predictions = archi(images)
                predicted_classes = torch.argmax(predictions, dim=1).cpu().numpy()
                match = predicted_classes == real_classes.cpu().numpy()
                corrects += sum(match)
                total += len(real_classes.cpu().numpy())

            res[mode] = corrects / total
        results[model_name] = res

    df = pd.DataFrame(results)
    print(df)
