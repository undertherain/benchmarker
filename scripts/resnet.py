#!/usr/bin/env python3
from run import run


def main():
    params = {}
    params["problem"] = "resnet50"
    params["mode"] = "training"
    for batch_size in range(1, 128):
        params["batch_size"] = batch_size * 2
        prob_size = params["batch_size"] * 4
        params["problem_size"] = f"{prob_size}"
        print(batch_size)
        params["gpus"] = "0"
        params["backend"] = "native"
        run(params)
        params.pop("gpus")
        params["backend"] = "DNNL"
        run(params)


if __name__ == "__main__":
    main()
