# python -m benchmarker --framework=pytorch --problem=ssd300 --problem_size=4 --batch_size=2 --mode=inference

import logging
import unittest

from ..helpers import run_module

logging.basicConfig(level=logging.DEBUG)


class PytorchSsd300Tests(unittest.TestCase):
    def setUp(self):
        self.args = [
            "benchmarker",
            "--problem=ssd300",
            "--framework=pytorch",
            "--problem_size=4",
            "--batch_size=2",
            "--mode=inference",
        ]

    def test_ssd300(self):
        run_module(*self.args)
