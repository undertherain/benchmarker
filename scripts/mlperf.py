#!/usr/bin/env python3
from run import run


def main():
    params = {}
    params["framework"] = "pytorch"
    params["problem"] = "resnet50"
    params["mode"] = "training"
    params["batch_size"] = 32
    prob_size = params["batch_size"] * 4
    params["problem_size"] = f"{prob_size}"
    params["gpus"] = "0"
#    params["backend"] = "DNNL"
    run(params)


if __name__ == "__main__":
    main()
