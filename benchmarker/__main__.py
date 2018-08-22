"""CLI entry point module"""

import argparse
from .benchmarker import run


def main():
    """CLI entry point function"""
    parser = argparse.ArgumentParser(description='Benchmark me up, Scotty!')
    parser.add_argument("--framework")
    parser.add_argument("--problem")
    parser.add_argument('--path_out', type=str, default="./logs")
    parser.add_argument('--gpus', default="")
    parser.add_argument('--size', default=None)
    parser.add_argument('--misc')

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
