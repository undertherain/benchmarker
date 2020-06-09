import logging
import unittest

from ..helpers import run_module

logging.basicConfig(level=logging.DEBUG)


class PytorchConv123dTests(unittest.TestCase):
    def setUp(self):
        self.args = [
            "benchmarker",
            "--framework=pytorch",
            "--batch_size=4",
            "--mode=inference",
        ]

    def test_conv1d(self):
        run_module(*self.args + ["--problem=conv1d", "--problem_size=4,4,4"])

    def test_conv2d(self):
        run_module(*self.args + ["--problem=conv2d", "--problem_size=4,4,4,4"])

    def test_conv3d(self):
        run_module(*self.args + ["--problem=conv3d", "--problem_size=4,4,4,4,4"])
