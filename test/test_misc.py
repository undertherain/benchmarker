import unittest

from .helpers import run_module


class MiscTests(unittest.TestCase):
    def setUp(self):
        self.name = "benchmarker"

    def test_no_framework(self):
        with self.assertRaises(Exception):
            run_module(self.name)

    def test_no_problem(self):
        with self.assertRaises(Exception):
            run_module(self.name, "--framework=pytorch")

    def test_bad_mode(self):
        with self.assertRaises(AssertionError):
            args = [
                "--framework=pytorch",
                "--problem=conv1d",
                "--problem_size=4,4,4",
                "--batch_size=4",
                "--mode=depeche",
            ]
            run_module(self.name, *args)
