import unittest
import os

test_rootpath = os.path.dirname(os.path.abspath(__file__))

from r3d3.experiment_launcher import main


class TestExperimentLauncher(unittest.TestCase):

    def test_main(self):
        launcher = main(f"{test_rootpath}/samples/sample_experiment.py")
        self.assertEqual(launcher.db.get_nb_experiments(), 4)
