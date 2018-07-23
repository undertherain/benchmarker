import argparse
from .benchmarker import run

#def run(framework: "Framework to test" = "numpy",
#         problem: "problem to solve" = "2048",
#         path_out: "path to store results" = "./logs",
#         gpus: "list of gpus to use" = "",
#         misc: "comma separated list of key:value pairs" = None
#         ):


def main():
    parser = argparse.ArgumentParser(description='Benchmark me up, Scotty!')
    parser.add_argument("--framework")
    parser.add_argument("--problem")
    parser.add_argument('--path_out', type=str, default="./logs")
    parser.add_argument('--gpus', default="")
    parser.add_argument('--misc')

    args = parser.parse_args()
    run(args.framework, args.problem, args.path_out, args.gpus, args.misc)


if __name__ == "__main__":
    main()
