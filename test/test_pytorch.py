import logging
import unittest

from .helpers import run_module

logging.basicConfig(level=logging.DEBUG)


class TorchTests(unittest.TestCase):
    def setUp(self):
        self.name = "benchmarker"
        self.imgnet_args = [
            "--framework=pytorch",
            "--problem_size=4",
            "--batch_size=2",
        ]

    def test_vgg16(self):
        run_module(self.name, "--problem=vgg16", *self.imgnet_args)

    def test_resnet50(self):
        run_module(self.name, "--problem=resnet50", *self.imgnet_args)

    def test_conv1d(self):
        run_module(
            self.name,
            "--framework=pytorch",
            "--problem=conv1d",
            "--problem_size=4,4,4",
            "--batch_size=4",
            "--mode=inference",
        )

    def test_conv2d(self):
        run_module(
            self.name,
            "--framework=pytorch",
            "--problem=conv2d",
            "--problem_size=4,4,4,4",
            "--batch_size=4",
            "--mode=inference",
        )

    def test_conv3d(self):
        run_module(
            self.name,
            "--framework=pytorch",
            "--problem=conv3d",
            "--problem_size=4,4,4,4,4",
            "--batch_size=4",
            "--mode=inference",
        )

    def test_lstm(self):
        run_module(
            self.name,
            "--framework=pytorch",
            "--problem=lstm",
            "--problem_size=4,4,4",
            "--batch_size=4",
            "--mode=inference",
        )
