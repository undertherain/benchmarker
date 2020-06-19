import logging
import unittest

from benchmarker.benchmarker import run

logging.basicConfig(level=logging.DEBUG)


class PytorchCNNTest(unittest.TestCase):
    def setUp(self):
        self.args = [
            "--problem=cnn2d_toy",
            "--framework=pytorch",
            "--problem_size=4",
            "--batch_size=2",
            "--nb_epoch=1",
        ]

    def test_cnn2d_inference(self):
        run(self.args + ["--mode=inference"])

    def test_cnn2d_training(self):
        run(self.args + ["--mode=training"])
