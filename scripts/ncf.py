#!/usr/bin/env python3
from run import fast_batches, run


def main():
    params = {}
    params["problem"] = "ncf"
    params["mode"] = "training"
    params["framework"] = "pytorch"
    params["nb_users"] = str(16*138000)
    params["nb_items"] = str(32*27000)
    for batch_size in fast_batches:
        params["batch_size"] = batch_size
        params["problem_size"] = 4 * batch_size
        print(batch_size)
        run(params)
    params["gpus"] = "0"
    for batch_size in fast_batches:
        params["batch_size"] = batch_size
        params["problem_size"] = 4 * batch_size
        print(batch_size)
        run(params)


if __name__ == "__main__":
    main()
