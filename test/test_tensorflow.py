import unittest
import logging
from .helpers import run_module

logging.basicConfig(level=logging.DEBUG)


class TensorflowTests(unittest.TestCase):
    # TODO(vatai): Run it quietly
    def setUp(self):
        self.name = "benchmarker"
        self.imgnet_args = [
            "--framework=tensorflow",
            "--problem_size=32",
            "--batch_size=16",
        ]

    def test_vgg16(self):
        run_module(self.name, "--problem=vgg16", *self.imgnet_args)

    def test_resnet50(self):
        run_module(self.name, "--problem=resnet50", *self.imgnet_args)
