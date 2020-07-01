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
            "--problem_size=2",
            "--batch_size=2",
            "--nb_epoch=1",
        ]

    def test_vgg16(self):
        run(["--problem=vgg16", *self.imgnet_args])

    def test_vgg16_inference(self):
        run(["--problem=vgg16", "--mode=inference", *self.imgnet_args])
