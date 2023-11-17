import logging
import unittest

from benchmarker.benchmarker import run

logging.basicConfig(level=logging.DEBUG)


class PytorchLstmTest(unittest.TestCase):
    def test_lstm(self):
        args = [
            "--framework=pytorch",
            "--problem=lstm",
            "--cnt_samples_per_epoch=4",
            "--sample_shape=4,4",
            "--batch_size=4",
            "--nb_epoch=1",
            "--mode=inference",
        ]
        run(args)
