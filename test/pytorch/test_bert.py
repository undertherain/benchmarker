import logging
import unittest

from ..helpers import run_module

logging.basicConfig(level=logging.DEBUG)


class PytorchBertTest(unittest.TestCase):
    def setUp(self):
        self.name = "benchmarker"

    def test_bert(self):
        run_module(
            self.name,
            "--framework=pytorch",
            "--problem=bert",
            "--problem_size=32,32",
            "--batch_size=8",
            "--nb_epoch=1",
            "--mode=inference",
            "--cnt_units=128",
            "--cnt_heads=4",
        )
