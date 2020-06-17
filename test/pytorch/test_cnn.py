import logging
import unittest

from ..helpers import run_module

logging.basicConfig(level=logging.DEBUG)


class PytorchCNNTest(unittest.TestCase):
    def setUp(self):
        self.name = "benchmarker"
        self.args = [
            "--problem=cnn2d_toy",
            "--framework=pytorch",
            "--problem_size=4",
            "--batch_size=2",
            "--nb_epoch=1",
        ]

    def test_cnn2d_inference(self):
        run_module(self.name, "--mode=inference", *self.args)

    def test_cnn2d_training(self):
        run_module(self.name, "--mode=training", *self.args)
