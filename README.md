This repository contains the code associated with the submission.

## 1) Getting started

### A) Preparatory steps

The main dependency of the project is dionysus, which require:
* Boost 1.55
* GCC 5.4 
(cf. [official doc from dionysus](https://pypi.org/project/dionysus/))


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
 
 ## 2) Launching experiments
 
 ### A) DB Setup
 
 All results are saved in a sqlite db using the library `r3d3`. 
 You need to set the value of the env variable `TDA_DB_PATH` in your bashrc.

### B) Running an experiment (recommended)

In r3d3, you basically create an experiment grid like [tda/experiments/thomas/embedding_separability_wl_gram.py](tda/experiments/thomas/embedding_separability_wl_gram.py) and then run it from the command line

```bash
r3d3-xp --experiment_file tda/experiments/global_plan.py
```

Under the hood it will call the binary n times for all the possibilities.

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

db.show_experiment(experiment_id=1566315142)
````
