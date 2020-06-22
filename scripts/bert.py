#!/usr/bin/env python3
from run import run_on_all_backends, fast_batches


def main():
    params = {}
    params["problem"] = "bert"
    params["mode"] = "training"
    for batch_size in fast_batches:
        params["batch_size"] = batch_size
        print(batch_size)
        run_on_all_backends(params)


if __name__ == "__main__":
    main()
