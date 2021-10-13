import getpass

if getpass.getuser() == "tricatte":
    tracking_uri = "file:///tda_code/mlflow"
else:
    tracking_uri = "https://mlflow.par.prod.crto.in"
