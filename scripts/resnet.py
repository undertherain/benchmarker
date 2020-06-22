#!/usr/bin/env python3
from run import run_on_all_backends, fast_batches


def main():
    params = {}
    params["problem"] = "resnet50"
    params["mode"] = "training"
    for params["batch_size"] in fast_batches:
        print(params["batch_size"])
        run_on_all_backends(params)


if __name__ == "__main__":
    main()
