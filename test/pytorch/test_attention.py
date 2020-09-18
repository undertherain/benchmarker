import logging
import unittest

from benchmarker.benchmarker import run

logging.basicConfig(level=logging.DEBUG)


class PytorchLstmTest(unittest.TestCase):
    def test_attention(self):
        args = [
            "--framework=pytorch",
            "--problem=attention",
            "--problem_size=2,2,4",
            "--cnt_heads=2",
            "--batch_size=1",
            "--nb_epoch=1",
            "--mode=inference",
        ]
        run(args)
