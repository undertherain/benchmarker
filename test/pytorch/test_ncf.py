import logging
import unittest

from benchmarker.benchmarker import run

logging.basicConfig(level=logging.DEBUG)


class PytorchNcfTest(unittest.TestCase):
    def setUp(self):
        self.args = [
            "--problem=ncf",
            "--framework=pytorch",
            "--cnt_samples_per_epoch=2",
            "--nb_epoch=1",
            "--batch_size=2",
        ]

    def test_ncf(self):
        run(self.args + ["--mode=training"])

    def test_ncf_inference(self):
        run(self.args + ["--mode=inference"])
