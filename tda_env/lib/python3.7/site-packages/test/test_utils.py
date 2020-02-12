import unittest
from functools import reduce
import typing
import json

from r3d3.utils import cartesian_product, namedtuple_to_dict


class FakeSubConfig(typing.NamedTuple):
    foo: str = "bar"


class FakeConfig(typing.NamedTuple):
    alpha: float = 0
    sub_conf: FakeSubConfig = FakeSubConfig()


class TestUtils(unittest.TestCase):

    def test_cartesian_product(self):
        my_grid = {
            "a": ["a1", "a2", "a3"],
            "b": ["b1", "b2"],
            "c": ["c1", "c2"]
        }

        all_variants = cartesian_product(my_grid)

        expected_nb_variants = reduce(lambda x, y: x * y, [len(v) for v in my_grid.values()])
        self.assertEqual(len(all_variants), expected_nb_variants)

    def test_namedtuple_to_json(self):
        self.assertEqual(
            json.dumps(namedtuple_to_dict(FakeConfig())),
            "{\"alpha\": 0, \"sub_conf\": {\"foo\": \"bar\"}}"
        )
