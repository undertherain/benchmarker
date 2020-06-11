import logging
import unittest

from ..helpers import run_module

logging.basicConfig(level=logging.DEBUG)


class PytorchResnet50Test(unittest.TestCase):
    def setUp(self):
        self.name = "benchmarker"
        self.imgnet_args = [
            "--problem=resnet50",
            "--framework=pytorch",
            "--problem_size=4",
            "--batch_size=2",
        ]

    def test_resnet50(self):
        run_module(self.name, "--mode=inference", *self.imgnet_args)

    def test_resnet50_inference(self):
        run_module(self.name, "--mode=inference", *self.imgnet_args)
