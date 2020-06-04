import logging
import unittest

from ..helpers import run_module

logging.basicConfig(level=logging.DEBUG)


class OpenCvRes10SsdTest(unittest.TestCase):
    def setUp(self):
        self.args = [
            "benchmarker",
            "--framework=opencv",
            "--problem=res10ssd",
            "--problem_size=4",
            "--batch_size=1",
            "--mode=inference",
        ]

    def test_res10ssd(self):
        run_module(*self.args)
