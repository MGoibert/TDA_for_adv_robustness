# Adversarial detection

## 1) Getting started

### A) Preparatory steps (on a criteo server, gpu, mozart)

All the code is compatible with the Criteo OS (i.e. the OS available either on Mozart or on the GPUs) except from dionysus, which required some additional setup.

When you install dionysus, you need two ingredients that are not available by default:
* Boost 1.55 (Criteo OS uses 1.53)
* GCC 5.4 (Criteo OS uses 4.8)
(cf. [official doc from dionysus](https://pypi.org/project/dionysus/))

#### Boost 1.55

To have boost available just follow these steps:
````bash
mkdir boost_155 && cd boost_155
wget http://sourceforge.net/projects/boost/files/boost/1.55.0/boost_1_55_0.tar.gz
tar xvzf boost_1_55_0.tar.gz
cd boost_1_55_0
pwd
````

The put the obtained path in the env variable `BOOST_ROOT` (in your bashrc).

#### GCC 5.4

This is a bit more tricky, we need two packages that have been recently added to chef at my request:
`devtoolset` and `cmake`. If they are not available on your machine and you are in mozart, you can install them directly.
 Otherwise, ping #gpu.
 
Then just run `scl enable devtoolset-8 bash` and now you are ready for the `pip install -e .`.


### ripser
`ripser` is an alternative to `dionysus`. It has the advantage that it does need the user
to provide manually precomputed filtrations. Installation of `ripser` via `pip` doesn't work, so we use `conda` for that. Viz

```bash
conda install -c conda-forge ripser
```

### B) Setup

*  Create a venv with python 3.6 and activate it
```bash
$ python -m venv my_env
$ source ./my_env/bin/activate
```
* Install the project in the env (you should be in the root folder of the project)
```bash
$ pip install -e .
``` 
* Opt. if you want to work on the GNN part, you will need pytorch geometric. The installation process is explained on this page https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html.

### C) Code organization

All experiments should go into experiments/YOUR_NAME. An experiments is currently based on an executable binary file and one or several experiment grid that are pointing to this binary.

For instance
 * [tda/experiments/thomas/embedding_separability_binary_gram.py](tda/experiments/thomas/embedding_separability_binary_gram.py) is the main binary to compute AUC.
 * [tda/experiments/thomas/embedding_separability_pd_gram.py](tda/experiments/thomas/embedding_separability_pd_gram.py) is a experiment grid using this binary with Persistent Diagrams.
 * [tda/experiments/thomas/embedding_separability_wl_gram.py](tda/experiments/thomas/embedding_separability_wl_gram.py) is a experiment grid using this binary with Weisfeler-Lehman embeddings.
 
 When an experiment is needs to be fixed, it should be copied to experiments/finalized/. Experiments in finalized are not meant to be modified.
 
 ## 2) Launching experiments
 
 ### A) DB Setup
 
 All results are saved in a sqlite db using `r3d3`, a small lib that I have created to help ML experiments in general. 
 You need to set the value of the env variable `TDA_DB_PATH` in your bashrc.
 
 Note that the binary is responsible to call r3d3 (this is pretty explicit [here](tda/experiments/thomas/embedding_separability_binary_gram.py) for instance).

 ### B) Running the binary directly
 
 This is the most straightforward solution. For instance
 
 ```bash
$ python experiments/thomas/embedding_separability_binary_gram.py --noise 0.0 --node_labels none --hash_size 50 --height 5 --threshold 15000 --embedding_type WeisfeilerLehman --max_nb_processes 1
```

You can check the binary to see what arguments are expected. There are two arguments that are a bit special experiment_id and run_id:
* experiment_id is the id you want to give to a group of experiments
* run_id is an id within this group

### C) Running using r3d3 (recommended)


In r3d3, you basically create an experiment grid like [tda/experiments/thomas/embedding_separability_wl_gram.py](tda/experiments/thomas/embedding_separability_wl_gram.py) and then run it from the command line

```bash
r3d3-xp --experiment_file tda/experiments/thomas/embedding_separability_wl_gram.py
```

Under the hood it will create an experiment_id and run_ids for you, create the cartesian product of the grid and call the binary n times for all the possibilities.

The results will be stored in the r3d3.db database.

## 3) Analyzing experiments

You can easily explore the results using the `ExperimentDB` object from `r3d3`

````python
from r3d3.experiment_db import ExperimentDB
from tda.rootpath import db_path

db = ExperimentDB(db_path)

# Create a pandas dataframe containing all results
df = db.list_all_experiments()

# OR

# Helper function to focus on one experiment
# both params and metrics dict are used to extract
# nested values.
# For instance if metrics is something like
# { "train": {"auc": 0.9}, "test": {"auc": 0.7}}
# then you can easily extract your column by using
# train.auc and test.auc
db.show_experiment(
    experiment_id=1566315142,
    params={"foo": "foo"},
    metrics={
        "training_auc": "train.auc",
        "testing_auc": "test.auc"
    }
)
````

Remark: I personally use jupyter notebooks with [qgrid](https://github.com/quantopian/qgrid) to explore the results.

For example, this code explores the results of one experiment:

````python
from tda.rootpath import db_path
from r3d3.experiment_db import ExperimentDB
import qgrid
import pandas as pd
import numpy as np

db_path = 'my/path/to/r3d3.db'
my_db = ExperimentDB(db_path)

# Print the dataframe of all experiments
my_db.list_all_experiments()

# Show the interesting results for one experiment (exp_id = 1588797205)
db.show_experiment(exp,
        params={
            "architecture": "architecture",
            "embedding_type": "embedding_type",
            "threshold": "thresholds",
            "attack_type": "attack_type"
        },
        metrics={
            "supervised_metrics": "supervised_metrics",
            "unsupervised_metrics": "unsupervised_metrics",
            "name" : "name"
        })
````
