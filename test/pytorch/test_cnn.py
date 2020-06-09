import logging
import unittest

from ..helpers import run_module

logging.basicConfig(level=logging.DEBUG)


class PytorchCNNTest(unittest.TestCase):
    def setUp(self):
        self.name = "benchmarker"
        self.imgnet_args = [
            "--framework=pytorch",
            "--problem_size=4",
            "--batch_size=2",
        ]

    def test_vgg16(self):
        run_module(self.name, "--problem=cnn2d_toy", *self.imgnet_args)
