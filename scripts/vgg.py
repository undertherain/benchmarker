#!/usr/bin/env python3
from run import run_on_all_backends, fast_batches


def main():
    params = {}
    params["problem"] = "vgg16"
    params["mode"] = "training"
    for batch_size in fast_batches:
        params["batch_size"] = batch_size * 2
        prob_size = params["batch_size"] * 4
        params["problem_size"] = f"{prob_size}"
        print(batch_size)
        run_on_all_backends(params)


if __name__ == "__main__":
    main()
