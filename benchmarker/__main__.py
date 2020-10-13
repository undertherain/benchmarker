"""CLI entry point module"""

import sys

from .benchmarker import run
from benchmarker.util import abstractprocess


def main():
    """CLI entry point function"""
    run(sys.argv[1:])
    # put corrected args to the command
    # run in a loop to collect counters
    command = "run benchmarker"
    proc = abstractprocess.Process("local", command=command)
    result = proc.get_output()
    # add to results
    # save


if __name__ == "__main__":
    main()
