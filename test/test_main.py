import unittest
from benchmarker.modules import do_numpy
import logging

logging.basicConfig(level=logging.DEBUG)


class Tests(unittest.TestCase):

    def test_main(self):
        print("hi")
