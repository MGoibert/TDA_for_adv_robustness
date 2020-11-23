import os
from pathlib import Path

rootpath = "{}/..".format(os.path.dirname(os.path.abspath(__file__)))

_default_model_dir = f"{rootpath}/trained_models"
model_dir = os.environ.get("TDA_MODEL_DIR", _default_model_dir)
Path(model_dir).mkdir(parents=True, exist_ok=True)
