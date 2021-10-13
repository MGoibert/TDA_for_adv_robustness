import getpass

if getpass.getuser() == "tricatte":
    tracking_uri = "/Users/tricatte/mlflow"
elif getpass.getuser() == "ec2-user":
    tracking_uri = "file:///tda_code/mlflow"
else:
    tracking_uri = "https://mlflow.par.prod.crto.in"
