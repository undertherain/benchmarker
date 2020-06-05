import logging
import os
import unittest

from ..helpers import run_module

logging.basicConfig(level=logging.DEBUG)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class XceptionTests(unittest.TestCase):
    def setUp(self):
        self.name = "benchmarker"
        self.imgnet_args = [
            "--framework=tensorflow",
            "--problem_size=4",
            "--batch_size=2",
        ]

    def test_xception(self):
        run_module(self.name, "--problem=xception", *self.imgnet_args)
