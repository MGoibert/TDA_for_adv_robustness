This repository contains the code associated with the submission.

## 1) Getting started

### A) Setup

*  Create a venv with python 3.6 and activate it
```bash
$ python -m venv my_env
$ source ./my_env/bin/activate
```
* Install the project in the env (you should be in the root folder of the project)
```bash
$ pip install -e .
``` 
 
 Note that we use dionysus to compute the persistent diagrams. This library requires boost (see the official doc)
 
## 2) Launching detection experiments

To launch the detection experiments, there are three main scripts

* `tda/experiments/lid/lid_binary.py` to launch the LID detector
* `tda/experiments/mahalanobis/mahalanobis_binary.py` to launc the Mahalanobis detector
* `tda/experiments/ours/our_binary.py` to launch our detector

Some arguments are common to all scripts, some others are specific (hyperparameters of the methods).

### A) Common arguments

All our scripts have some common arguments:
* attack_type (can be typically FGSM, PGD, CW, BOUNDARY)
* dataset (can be MNIST, SVHN, CIFAR10)
* architecture (see below)
* epochs (number of epochs for the architecture ; see below)
* dataset_size (we use 500 by default)

Note that all combinations of dataset / architectures are not possible due to the different shapes of the images. In the paper, we use the following:

| Dataset | Architecture | Nb Epochs  |
| --- |:---:| ---:|
| MNIST      | mnist_lenet | 50 |
| FashionMNIST | fashion_mnist_lenet | 100 |
| SVHN | svhn_lenet | 300 |
| CIFAR10 | cifar_lenet | 300 |
| CIFAR10 | cifar_resnet_1 | 100 |

For all the combinations, the trained models are provided in the git repository so the code won't have to retrain them. Due to the size of the cifar_resnet_1 model, it wasn't included in git and it will be automatically retrained if you launch it.

### B) Our method

Our hyperparameters are all discussed in the paper but here's a complementary table

| Parameter | Description |
| --- |---|
| embedding_type | PersistentDiagram or RawGraph |
| kernel_type | SlicedWassertein (to be used with Persistent Diagram) or RBF (to be used with RawGraph) |
| thresholds | The maximum quantile to keep (see paper) expressed as layeridx:value_layeridx:value_ ...
| sigmoidize | to remove |
| raw_graph_pca | The dimension of the PCA to be used with RawGraph (deactivated by default) |

Example:

```bash
python tda/experiments/ours/our_binary.py \
    --attack_type PGD \
    --architecture cifar_resnet_1 \
    --dataset CIFAR10 \
    --epochs 100 \
    --embedding_type PersistentDiagram \
    --kernel_type SlicedWasserstein \
    --thresholds 0.05 \
    --sigmoidize True
```

### C) LID Baseline

The specific arguments for LID are the `batch_size` and the `perc_of_nn` (percentage of nearest neighbors in the batch) for the LID estimation.

Example:

```bash
python tda/experiments/lid/lid_binary.py \
    --attack_type PGD \
    --architecture cifar_resnet_1 \
    --dataset CIFAR10 \
    --epochs 100 \
    --perc_of_nn 0.3 \
    --batch_size 100 
```

### D) Mahalanobis Baseline

The specific arguments for Mahalanobis are `number_of_samples_for_mu_sigma` (number of samples used for the estimation) and `preproc_epsilon`.

Example:

```bash
python tda/experiments/mahalanobis/mahalanobis_binary.py \
    --attack_type PGD \
    --architecture cifar_lenet \
    --dataset CIFAR10 \
    --epochs 300 \
    --number_of_samples_for_mu_sigma 500 \
    --preproc_epsilon 0.01
```

