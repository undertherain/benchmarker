import logging
import unittest

from benchmarker.benchmarker import run

logging.basicConfig(level=logging.DEBUG)


class PytorchConv123dTests(unittest.TestCase):
    def setUp(self):
        self.args = [
            "--framework=pytorch",
            "--batch_size=2",
            "--nb_epoch=1",
            "--mode=inference",
        ]

    def test_conv1d(self):
        run(self.args + ["--problem=conv1d", "--problem_size=2,4,4"])

    def test_conv2d(self):
        run(self.args + ["--problem=conv2d", "--problem_size=2,4,4,4"])

    def test_conv3d(self):
        run(self.args + ["--problem=conv3d", "--problem_size=2,4,4,4,4"])
