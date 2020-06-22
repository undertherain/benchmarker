#!/usr/bin/env python3
from run import run, fast_batches


def main():
    params = {}
    params["framework"] = "pytorch"
    params["problem"] = "bert"
    params["mode"] = "training"
    for batch_size in fast_batches:
        params["batch_size"] = batch_size * 2
        prob_size = params["batch_size"] * 4
        params["problem_size"] = f"{prob_size}, 128"
        print(batch_size)
        params["gpus"] = "0"
        params["backend"] = "native"
        run(params)
        params.pop("gpus")
        params["backend"] = "DNNL"
        print(batch_size)
        run(params)


if __name__ == "__main__":
    main()
