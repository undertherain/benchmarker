"""
This is helper module to profile the whole package
in Python 3.7 profiling modules from command line will be supported
and this module will no longer be needed
"""

import cProfile
from .__main__ import main

if __name__ == "__main__":
    cProfile.run("main()", filename=".beholder.cprofile")
