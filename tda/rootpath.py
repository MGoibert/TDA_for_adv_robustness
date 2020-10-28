import os
from pathlib import Path

rootpath = "{}/..".format(os.path.dirname(os.path.abspath(__file__)))

_default_db_path = f"{rootpath}/r3d3.db"
db_path = os.environ.get("TDA_DB_PATH", _default_db_path)

_default_model_dir = f"{rootpath}/trained_models"
model_dir = os.environ.get("TDA_MODEL_DIR", _default_model_dir)
Path(model_dir).mkdir(parents=True, exist_ok=True)
