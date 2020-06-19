import unittest

from benchmarker.benchmarker import run


class MiscTests(unittest.TestCase):
    def test_no_framework(self):
        with self.assertRaises(Exception):
            run([])

    def test_no_problem(self):
        with self.assertRaises(Exception):
            run(["--framework=pytorch"])

    def test_bad_mode(self):
        with self.assertRaises(AssertionError):
            args = [
                "--framework=pytorch",
                "--problem=conv1d",
                "--problem_size=4,4,4",
                "--batch_size=4",
                "--mode=depeche",
            ]
            run(args)
