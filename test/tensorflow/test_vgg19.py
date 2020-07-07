import logging
import os
import unittest

from benchmarker.benchmarker import run

logging.basicConfig(level=logging.DEBUG)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class TensorflowTests(unittest.TestCase):
    def setUp(self):
        self.imgnet_args = [
            "--framework=tensorflow",
            "--problem_size=4",
            "--batch_size=2",
            "--nb_epoch=1",
        ]

    def test_vgg19(self):
        run(["--problem=vgg19", *self.imgnet_args])
