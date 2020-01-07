# -*- coding: utf-8 -*-
"""NumPy support.
"""

import argparse
from timeit import default_timer as timer
import numpy as np


class Benchmark():
    def __init__(self, params, remaining_args=None):
        self.params = params
        parser = argparse.ArgumentParser(description='Benchmark GEMM operations')
        #parser.add_argument('--size', default=None)
        #args = parser.parse_args(remaining_args)
        #params["mode"] = args.mode
        #params["path_out"] = os.path.join(params["path_out"], params["mode"])

        # TODO: read size from args
        # TODO: add float type as arg

    def run(self):
        params = self.params
        if params["problem"]["name"] != "gemm":
            raise Exception("only gemm problem is defined for numpy")
        if "nb_gpus" in params:
            if params["nb_gpus"] > 0:
                raise Exception("Numpy framework does not work with GPU back-end")

        M = 16 * 400
        N = 16 * 400
        K = 16 * 400
        a = np.random.random((M, K))
        b = np.random.random((K, N))
        c = np.random.random((K, K))

        nb_epoch = 2

        time_start = timer()
        for _ in range(nb_epoch):
            c = a @ b + c
        time_end = timer()
        gflop = (2.0 * M * N * K) / (1024 ** 3)
        #print(f"GFlop: {gflop:.5f}")
        elapsed_time = (time_end - time_start) / nb_epoch
        #print(f"time: {elapsed_time:.5f}")
        #print(f"GFlop/sec: {gflop / elapsed_time:.5f}")

        params["time"] = elapsed_time
        params["GFLOP"] = gflop
        params["GFLOP/sec"] = gflop / elapsed_time
        return params
