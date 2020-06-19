import logging
import unittest

from benchmarker.benchmarker import run

logging.basicConfig(level=logging.DEBUG)


class PytorchVgg16Test(unittest.TestCase):
    def setUp(self):
        self.args = [
            "--problem=vgg16",
            "--framework=pytorch",
            "--problem_size=4",
            "--batch_size=2",
            "--nb_epoch=1",
        ]

    def test_vgg16(self):
        run(self.args + ["--mode=training"])

    def test_vgg16_inference(self):
        run(self.args + ["--mode=inference"])
