import logging
import unittest

from ..helpers import run_module

logging.basicConfig(level=logging.DEBUG)


class PytorchLstmTest(unittest.TestCase):
    def test_lstm(self):
        run_module(
            "benchmarker",
            "--framework=pytorch",
            "--problem=lstm",
            "--problem_size=4,4,4",
            "--batch_size=4",
            "--nb_epoch=1",
            "--mode=inference",
        )
