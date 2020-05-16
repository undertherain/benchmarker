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
        # vatai: Strange that only "--problem conv1d",
        # "--problem=conv1d" doesn't.
        run_module(
            self.name,
            "--framework=pytorch",
            "--problem conv1d",
            "--problem_size='(4, 4, 4)'",
            "--batch_size=64",
            "--mode=inference",
        )

    def test_conv2d(self):
        run_module(
            self.name,
            "--framework=pytorch",
            "--problem conv1d",
            "--problem_size='(4, 4, 4, 4)'",
            "--batch_size=64",
            "--mode=inference",
        )

    def test_conv2d_1(self):
        run_module(
            self.name,
            "--framework=pytorch",
            "--problem conv1d_1",
            "--problem_size='(4, 4, 4, 4)'",
            "--batch_size=64",
            "--mode=inference",
        )

    def test_conv2d_2(self):
        run_module(
            self.name,
            "--framework=pytorch",
            "--problem conv1d_2",
            "--problem_size='(4, 4, 4, 4)'",
            "--batch_size=64",
            "--mode=inference",
        )

    def test_conv3d(self):
        run_module(
            self.name,
            "--framework=pytorch",
            "--problem conv1d_2",
            "--problem_size='(4, 4, 4, 4, 4)'",
            "--batch_size=64",
            "--mode=inference",
        )

    def test_conv3d_1(self):
        run_module(
            self.name,
            "--framework=pytorch",
            "--problem conv1d_2",
            "--problem_size='(4, 4, 4, 4, 4)'",
            "--batch_size=64",
            "--mode=inference",
        )
