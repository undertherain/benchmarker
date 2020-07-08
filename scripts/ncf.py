#!/usr/bin/env python3
from run import fast_batches, run


def main():
    params = {}
    params["problem"] = "bert"
    params["mode"] = "training"
    params["framework"] = "pytorch"
    for batch_size in fast_batches:
        params["batch_size"] = batch_size
        print(batch_size)
        run(params)


if __name__ == "__main__":
    main()
