"""CLI entry point module"""

import sys

from .benchmarker import run


def main():
    """CLI entry point function"""
    run(sys.argv[1:])


if __name__ == "__main__":
    main()
