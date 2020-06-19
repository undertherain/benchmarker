import logging
import unittest

from benchmarker.benchmarker import run

logging.basicConfig(level=logging.DEBUG)


class OpenCvRes10SsdTest(unittest.TestCase):
    def setUp(self):
        self.args = [
            "--framework=opencv",
            "--problem=res10ssd",
            "--problem_size=4",
            "--batch_size=1",
            "--mode=inference",
        ]

    def test_res10ssd(self):
        run(self.args)
