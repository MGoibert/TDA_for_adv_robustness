# Adversarial detection

## Getting started

### Setup

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


### Code organization

All experiments should go into experiments/YOUR_NAME. An experiments is currently based on an executable binary file and one or several experiment grid that are pointing to this binary.

For instance
 * [tda/experiments/thomas/embedding_separability_binary_gram.py](tda/experiments/thomas/embedding_separability_binary_gram.py) is the main binary to compute AUC.
 * [tda/experiments/thomas/embedding_separability_pd_gram.py](tda/experiments/thomas/embedding_separability_pd_gram.py) is a experiment grid using this binary with Persistent Diagrams.
 * [tda/experiments/thomas/embedding_separability_wl_gram.py](tda/experiments/thomas/embedding_separability_wl_gram.py) is a experiment grid using this binary with Weisfeler-Lehman embeddings.
 
 When an experiment is needs to be fixed, it should be copied to experiments/finalized/. Experiments in finalized are not meant to be modified.
 
 ## Launching experiments
 
 ### Running the binary directly (not recommended)
 
 This is the most straightforward solution. For instance
 
 ```bash
$ python experiments/thomas/embedding_separability_binary_gram.py --noise 0.0 --node_labels none --hash_size 50 --height 5 --threshold 15000 --embedding_type WeisfeilerLehman --max_nb_processes 1
```

You can check the binary to see what arguments are expected. There are two arguments that are a bit special experiment_id and run_id:
* experiment_id is the id you want to give to a group of experiments
* run_id is an id within this group

All the results will be stored in an sqlite database. To do so, the binary should use the `ExperimentDB` object from `r3d3` (pretty explicit in the existing binary).

### Running using r3d3 (recommended)

r3d3 is a small lib that I have done to help ML experiments in general. 
In r3d3, you basically create an experiment grid like [tda/experiments/thomas/embedding_separability_wl_gram.py](tda/experiments/thomas/embedding_separability_wl_gram.py) and then run it from the command line

```bash
r3d3-xp --experiment_file tda/experiments/thomas/embedding_separability_wl_gram.py
```

Under the hood it will create an experiment_id and run_ids for you, create the cartesian product of the grid and call the binary n times for all the possibilities.

## Analyzing experiments

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