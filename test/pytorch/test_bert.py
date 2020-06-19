import logging
import unittest

from ..helpers import run_module

logging.basicConfig(level=logging.DEBUG)


class PytorchBertTest(unittest.TestCase):
    def setUp(self):
        self.name = "benchmarker"

    def test_bert_inference(self):
        run_module(
            self.name,
            "--framework=pytorch",
            "--problem=bert",
            "--problem_size=4,8",
            "--batch_size=2",
            "--nb_epoch=1",
            "--mode=inference",
        )

    def test_bert_training(self):
        run_module(
            self.name,
            "--framework=pytorch",
            "--problem=bert",
            "--problem_size=4,8",
            "--batch_size=2",
            "--nb_epoch=1",
            "--mode=training",
        )
