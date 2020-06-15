import logging
import unittest

from ..helpers import run_module

logging.basicConfig(level=logging.DEBUG)


class PytorchVgg16Test(unittest.TestCase):
    def setUp(self):
        self.name = "benchmarker"
        self.imgnet_args = [
            "--problem=vgg16",
            "--framework=pytorch",
            "--problem_size=4",
            "--batch_size=2",
        ]

    def test_vgg16(self):
        run_module(self.name, "--mode=training", *self.imgnet_args)

    def test_vgg16_inference(self):
        run_module(self.name, "--mode=inference", *self.imgnet_args)
