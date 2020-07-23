import logging
import os
import unittest

from benchmarker.benchmarker import run

logging.basicConfig(level=logging.DEBUG)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class CosmoflowTests(unittest.TestCase):
    def setUp(self):
        self.args = [
            "--problem=cosmoflow",
            "--framework=pytorch",
            "--problem_size=1",
            "--batch_size=1",
            "--nb_epoch=1",
        ]

    def test_cosmoflow(self):
        run(self.args)

    def test_cosmoflow_inference(self):
        run(self.args + ["--mode=inference"])
