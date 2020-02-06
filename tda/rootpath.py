import os

rootpath = "{}/..".format(os.path.dirname(os.path.abspath(__file__)))

_default_db_path = f"{rootpath}/r3d3.db"
db_path = os.environ.get('TDA_DB_PATH', _default_db_path)