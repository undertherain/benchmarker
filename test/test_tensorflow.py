import unittest
import logging
from .helpers import run_module

import os
from io import StringIO
from contextlib import redirect_stderr, redirect_stdout

logging.basicConfig(level=logging.DEBUG)


class TensorflowTests(unittest.TestCase):
    # TODO(vatai): Run it quietly
    def setUp(self):
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        self.name = "benchmarker"
        self.imgnet_args = [
            "--framework=tensorflow",
            "--problem_size=32",
            "--batch_size=16",
        ]

    def test_vgg16(self):
        sio = StringIO()
        with redirect_stdout(sio), redirect_stderr(sio):
            run_module(self.name, "--problem=vgg16", *self.imgnet_args)

    def test_resnet50(self):
        sio = StringIO()
        with redirect_stdout(sio), redirect_stderr(sio):
            run_module(self.name, "--problem=resnet50", *self.imgnet_args)
