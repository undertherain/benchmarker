import logging
import os
import unittest

from ..helpers import run_module

logging.basicConfig(level=logging.DEBUG)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class TensorflowTests(unittest.TestCase):
    def setUp(self):
        self.name = "benchmarker"
        self.imgnet_args = [
            "--framework=tensorflow",
            "--problem_size=2",
            "--batch_size=2",
            "--nb_epoch=1",
        ]

    def test_vgg16(self):
        run_module(self.name, "--problem=vgg16", *self.imgnet_args)
