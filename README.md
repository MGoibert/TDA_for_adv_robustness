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
* `tda/experiments/mahalanobis/mahalanobis_binary.py` to launch the Mahalanobis detector
* `tda/experiments/ours/our_binary.py` to launch our detector

Some arguments are common to all scripts, some others are specific (hyperparameters of the methods).

### A) Common arguments

All our scripts have some common arguments:
* `attack_type` (can be typically FGSM, PGD, CW, BOUNDARY)
* `dataset` (can be MNIST, Fashion MNIST, SVHN, CIFAR10)
* `architecture` (see below)
* `epochs` (number of epochs for the architecture ; see below)
* `dataset_size` (we use 500 by default)
* `all_epsilons` (list of l-inf perturbations only for PGD and FGSM, separated by ;)

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

Our script takes the following arguments

| Parameter | Type | Description |
| --- |---| ---|
| embedding_type | str | PersistentDiagram or RawGraph |
| kernel_type | str | SlicedWassertein (to be used with Persistence Diagram) or RBF (to be used with RawGraph) |
| thresholds | str | the thresholds to apply to the induced graph (see below)
| threshold_strategy | str | the threshold strategy to apply |
| raw_graph_pca | int | The dimension of the PCA to be used with RawGraph (deactivated by default) |

The format for the threshods argument is 

`idx1:qmin1:qmax1_idx2:qmin2:qmax2_....`

where `idx1`, `idx2`, ... are the indices of the layers we want to use for the induced graph.
For each layer, we keep the edges whose value is in between the quantiles `qmin` and `qmax`.

The value used for the thresholding depends on the `threshold_strategy` parameter:
* if `UnderoptimizedMagnitudeIncrease` then we use the magnitude increase criteria
* if `UnderoptimizedLargeFinal` we use the absolute value of the weight

*Example 1: using 30% underoptimized edges with Magnitude Increase criteria on CIFAR10*

```bash
python tda/experiments/ours/our_binary.py \
    --attack_type PGD \
    --architecture cifar_resnet_1 \
    --dataset CIFAR10 \
    --epochs 100 \
    --embedding_type PersistentDiagram \
    --kernel_type SlicedWasserstein \
    --thresholds 39:0.0:0.3_43:0.0:0.3 \
    --threshold_strategy UnderoptimizedMagnitudeIncrease
```

(the layer indices of the last conv. and linear layers are respectively 39 and 43)

*Example 2: using 30% edges with largest value on CIFAR10*

we modify the previous command with 

```
    --thresholds 39:0.7:1.0_43:0.7:1.0 \
    --threshold_strategy UnderoptimizedLargeFinal
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

## 3) How to reproduce main figures ?

### Example 1: Figure 2: Influence of q

Our curve to be run for different values of q between 0 and 1:
```bash
python tda/experiments/ours/our_binary.py \
    --attack_type PGD \
    --architecture cifar_resnet_1 \
    --dataset CIFAR10 \
    --epochs 100 \
    --embedding_type PersistentDiagram \
    --kernel_type SlicedWasserstein \
    --thresholds 39:0.0:q_43:0.0:q \
    --threshold_strategy UnderoptimizedMagnitudeIncrease
```

### Example 2: Figure 5: Edge selection method

Total of 6 experiments to launch

* MNIST
    - Magnitude Increase
        ```
        python tda/experiments/ours/our_binary.py \
            --attack_type PGD \
            --architecture mnist_lenet \
            --dataset MNIST \
            --epochs 50 \
            --embedding_type PersistentDiagram \
            --kernel_type SlicedWasserstein \
            --thresholds "0:0.0:0.025_2:0.0:0.025_4:0.0:0.025_6:0.0:0.025" \
            --threshold_strategy UnderoptimizedMagnitudeIncrease
        ```
    - Small-valued edges
        
        same as above but with
        ```
            --thresholds "0:0.0:0.025_2:0.0:0.025_4:0.0:0.025_6:0.0:0.025" \
            --threshold_strategy UnderoptimizedLargeFinal
        ```
    - Large-valued edges
        
        same as above but with
        ```
            --thresholds "0:0.975:1.0_2:0.975:1.0_4:0.975:1.0_6:0.975:1.0" \
            --threshold_strategy UnderoptimizedLargeFinal
        ```
* CIFAR10
    - Magnitude Increase
        ```
        python tda/experiments/ours/our_binary.py \
            --attack_type PGD \
            --architecture cifar_resnet_1 \
            --dataset CIFAR10 \
            --epochs 100 \
            --embedding_type PersistentDiagram \
            --kernel_type SlicedWasserstein \
            --thresholds "39:0.0:0.3_43:0.0:0.3" \
            --threshold_strategy UnderoptimizedMagnitudeIncrease
        ```
    - Small-valued edges
        
        same as above but with
        ```
            --thresholds "39:0.0:0.3_43:0.0:0.3" \
            --threshold_strategy UnderoptimizedLargeFinal
        ```
    - Large-valued edges
        
        same as above but with
        ```
            --thresholds "39:0.7:1.0_43:0.7:1.0" \
            --threshold_strategy UnderoptimizedLargeFinal
        ```
      
      

## 4) Layer indices

Here we detail the layer indices of our architectures.

### A) LeNet

0. Conv(1 -> 10, 5)
1. MaxPool2dLayer(2)
2. Conv(10 -> 20, 5)
3. MaxPool2dLayer(2)
4. Linear(320 -> 50)
5. DropOut()
6. Linear(50 -> 10)
7. SoftMaxLayer()

### B) ResNet

	
0.	Conv(3->64, 3)
1.	BatchNorm2d(ReLU)
2.	Conv(64->64, 3) # Block 1/a
3.	BatchNorm2d(ReLU)
4.	Conv(64->64, 3)
5.	BatchNorm2d
6.	ReLU	
7.	Conv(64->64, 3)  # Block 1/b
8.	BatchNorm2d(ReLU)
9.	Conv(64->64, 3)
10.	BatchNorm2d
11.	ReLU	
12.	Conv(64->128, 3) # Block 2/a
13.	BatchNorm2d(ReLU)
14.	Conv(128->128, 3)
15.	BatchNorm2d
16.	ReLU	
17.	Conv(128->128, 3) # Block 2/b
18.	BatchNorm2d(ReLU)
19.	Conv(128->128, 3)
20.	BatchNorm2d
21.	ReLU	
22.	Conv(128->256, 3) # Block 3/a
23.	BatchNorm2d(ReLU)
24.	Conv(256->256, 3)
25.	BatchNorm2d
26.	ReLU	
27.	Conv(256->256, 3) # Block 3/b
28.	BatchNorm2d(ReLU)
29.	Conv(256->256, 3)
30.	BatchNorm2d
31.	ReLU
32.	Conv(256->512, 3) # Block 4/a
33.	BatchNorm2d(ReLU)
34.	Conv(512->512, 3)
35.	BatchNorm2d
36.	ReLU	
37.	Conv(512->512, 3) # Block 4/b
38.	BatchNorm2d(ReLU)
39.	Conv(512->512, 3)
40.	BatchNorm2d
41.	ReLU	
42.	AvgPool2d # Skips
43.	Linear(512->10)
44.	Softmax
