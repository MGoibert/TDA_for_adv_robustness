import unittest
import typing
import os

from r3d3.experiment_db import ExperimentDB
import tempfile


class FakeConfig(typing.NamedTuple):
    alpha: float = 0


class TestExperimentDB(unittest.TestCase):

    def test_nb_experiments(self):

        db_path = tempfile.mkstemp()[1]

        db = ExperimentDB(db_path=db_path)
        db.init_experiment_table(drop=True)

        self.assertEqual(db.get_nb_experiments(), 0)

        db.add_experiment(
            experiment_id=0,
            run_id=1,
            config=FakeConfig()
        )
        self.assertEqual(db.get_nb_experiments(), 1)
        df = db.list_all_experiments()
        self.assertEqual(len(df), 1)

        os.remove(db_path)
