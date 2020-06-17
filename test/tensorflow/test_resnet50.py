import logging
import os
import unittest

from ..helpers import run_module

logging.basicConfig(level=logging.DEBUG)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class Resnet50Tests(unittest.TestCase):
    def setUp(self):
        self.name = "benchmarker"
        self.imgnet_args = [
            "--framework=tensorflow",
            "--problem_size=2",
            "--batch_size=2",
            "--platform_info=no"
        ]

    def test_resnet50(self):
        run_module(self.name, "--problem=resnet50", *self.imgnet_args)
