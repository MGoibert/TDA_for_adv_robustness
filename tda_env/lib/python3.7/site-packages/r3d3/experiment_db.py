import datetime
import json
import logging
import os
import sqlite3
import typing
from contextlib import contextmanager

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("[ExperimentDB]")


class ExperimentDB(object):

    def __init__(self, db_path):
        self.db_path = db_path

    @contextmanager
    def db_cursor(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            yield cursor
        except Exception as e:
            logger.error("Error: {}".format(e))
            conn.rollback()
        else:
            conn.commit()

    # Initialize experiments table
    def init_experiment_table(self, drop=False):
        with self.db_cursor() as cur:
            if drop:
                cur.execute("DROP TABLE IF EXISTS experiments")

            cur.execute('''CREATE TABLE IF NOT EXISTS experiments
                         (
                         experiment_id integer,
                         run_id integer,
                         date text,
                         config text,
                         metrics text,
                         owner text,
                         PRIMARY KEY (experiment_id, run_id, owner)
                         )''')

    def get_nb_experiments(self):
        nb_experiments = 0

        with self.db_cursor() as cur:
            cur.execute("SELECT count(1) FROM experiments")
            nb_experiments = cur.fetchone()[0]

        return nb_experiments

    def add_experiment(self, experiment_id: int,
                       run_id: int,
                       config: typing.Dict):
        self.init_experiment_table(drop=False)

        with self.db_cursor() as cur:
            date = str(datetime.datetime.now().isoformat())

            cur.execute(f"""INSERT INTO experiments VALUES (
                {experiment_id},
                {run_id},
                '{date}',
                '{json.dumps(config)}',
                '',
                '{os.environ.get("USER", "unknown")}'
            )
            """)

    def update_experiment(self, experiment_id, run_id, metrics):
        with self.db_cursor() as cur:
            cur.execute(f"""
            UPDATE experiments
            SET metrics = '{json.dumps(metrics)}'
            WHERE run_id = '{run_id}' AND experiment_id = '{experiment_id}'""")

    def list_all_experiments(self):
        with self.db_cursor() as cur:
            ret = list()
            for row in cur.execute("SELECT * FROM experiments"):
                ret.append(row)

        return pd.DataFrame(
            data=ret,
            columns=[
                "experiment_id",
                "run_id",
                "date",
                "config",
                "metrics",
                "owner"])

    @staticmethod
    def parse_json(s):
        try:
            return json.loads(s)
        except:
            return dict()

    @staticmethod
    def recursive_get(d: typing.Dict, path: str):
        path_spl = path.split(".")
        res = d
        for elem in path_spl:
            res = res.get(elem, dict())

        if isinstance(res, dict) and len(res) == 0:
            return None

        return res

    def show_experiment(self, experiment_id: int, params: typing.Dict, metrics: typing.Dict):
        with self.db_cursor() as cur:
            ret = list()
            for row in cur.execute(f"SELECT * FROM experiments WHERE experiment_id = '{experiment_id}'"):
                ret.append(row)

        df = pd.DataFrame(
            data=ret,
            columns=[
                "experiment_id",
                "run_id",
                "date",
                "config",
                "metrics",
                "owner"])
        df["run_id"] = df["run_id"].apply(int)

        for name in params:
            df[name] = df["config"].apply(lambda s:
                                          ExperimentDB.recursive_get(
                                              ExperimentDB.parse_json(s), params[name]))
        for name in metrics:
            df[name] = df["metrics"].apply(lambda s:
                                           ExperimentDB.recursive_get(
                                               ExperimentDB.parse_json(s), metrics[name]))

        df.drop(columns=["experiment_id", "date", "metrics", "config", "owner"], inplace=True)

        return df
