import glob
import os
import tempfile
import requests
from PIL import Image
from tqdm.notebook import tqdm
import zipfile
import torchvision.transforms as transforms

from tda.rootpath import rootpath
from tda.devices import device

_url_source = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
_root = f"{rootpath}/tiny_image_net"
_real_root = f"{_root}/tiny-imagenet-200"

_default_trans = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.to(device)),
        transforms.Normalize((0.0,), (1.0,)),
    ]
)

def load_image(infilename):
    """This function loads an image into memory when you give it
       the path of the image
    """
    img = Image.open(infilename).convert("RGB")
    img.load()
    return img


def load_tiny_image_net_classes():
    with open(f"{_real_root}/words.txt") as f:
        label_to_name = dict([s.strip().split("\t") for s in f])
    all_labels = list(
        sorted([s.split("/")[-1] for s in glob.glob(f"{_real_root}/train/n*")])
    )
    all_names = [label_to_name.get(lab) for lab in all_labels]
    return all_labels, all_names


def load_tiny_image_net(transform=None, mode="train"):

    transform = transform or _default_trans

    if not os.path.exists(_root):
        print("Dataset not found. Downloading it...")
        os.mkdir(_root)
        with tempfile.NamedTemporaryFile() as fp:
            r = requests.get(_url_source, allow_redirects=True)
            fp.write(r.content)
            with zipfile.ZipFile(fp.name, "r") as zip_ref:
                zip_ref.extractall(_root)
    else:
        print("Found dataset...")

    all_labels, all_names = load_tiny_image_net_classes()

    samples = list()
    labels = list()

    if mode == "train":
        folders = glob.glob(f"{_real_root}/{mode}/n*")

        for folder in tqdm(folders):
            images = glob.glob(f"{folder}/images/*.JPEG")
            for image in images:
                img_data = load_image(image)
                img_data = transform(img_data)
                assert img_data.shape == (3, 64, 64)
                samples.append(img_data)
                labels.append(all_labels.index(folder.split("/")[-1]))

    elif mode == "test":

        with open(f"{_real_root}/val/val_annotations.txt") as f:
            img_to_label = [line.strip().split("\t") for line in f]
        img_to_label = dict([(l[0], l[1]) for l in img_to_label])

        images = glob.glob(f"{_real_root}/val/*/*.JPEG")

        for image in tqdm(images):
            img_data = load_image(image)
            img_data = transform(img_data)
            assert img_data.shape == (3, 64, 64)
            img_label = img_to_label.get(image.split("/")[-1])
            samples.append(img_data)
            labels.append(all_labels.index(img_label))

    return list(zip(samples, labels))
