import os

rootpath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
_default_db_path = None
if "USER" in os.environ:
    _default_db_path = f"/Users/{os.environ['USER']}/r3d3.db"
db_path = os.environ.get('TDA_DB_PATH', _default_db_path)
if db_path is None:
    db_path = "./r3d3.db"
