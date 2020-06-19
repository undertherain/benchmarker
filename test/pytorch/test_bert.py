import logging
import unittest

from benchmarker.benchmarker import run

logging.basicConfig(level=logging.DEBUG)


class PytorchBertTest(unittest.TestCase):
    def setUp(self):
        self.args = [
            "--framework=pytorch",
            "--problem=bert",
            "--problem_size=4,8",
            "--batch_size=2",
            "--nb_epoch=1",
        ]

    def test_bert_inference(self):
        run(self.args + ["--mode=inference"])

    def test_bert_training(self):
        run(self.args + ["--mode=training"])
