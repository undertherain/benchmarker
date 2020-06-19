import logging
import os
import unittest

from benchmarker.benchmarker import run

logging.basicConfig(level=logging.DEBUG)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class Resnet50Tests(unittest.TestCase):
    def setUp(self):
        self.imgnet_args = [
            "--framework=tensorflow",
            "--problem_size=2",
            "--batch_size=2",
            "--nb_epoch=1",
        ]

    def test_resnet50(self):
        run(["--problem=resnet50"] + self.imgnet_args)

    def test_resnet50_inference(self):
        run(["--problem=resnet50", "--mode=inference"] + self.imgnet_args)
