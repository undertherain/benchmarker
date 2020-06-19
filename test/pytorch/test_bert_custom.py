import logging
import unittest

from benchmarker.benchmarker import run

logging.basicConfig(level=logging.DEBUG)


class PytorchBertTest(unittest.TestCase):
    def setUp(self):
        self.args = [
            "--framework=pytorch",
            "--problem=bert_custom",
            "--problem_size=32,32",
            "--batch_size=8",
            "--nb_epoch=1",
            "--mode=inference",
            "--cnt_units=128",
            "--cnt_heads=4",
        ]

    def test_bert(self):
        run(self.args)
