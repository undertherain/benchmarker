# -*- coding: utf-8 -*-
"""NumPy support.
"""

from timeit import default_timer as timer
import numpy as np
import os
from .i_gemm import IGEMM


class Benchmark(IGEMM):
    def __init__(self, params, remaining_args=None):
        super().__init__(params, remaining_args)

    def run(self):
        params = self.params
        if params["problem"]["name"] != "gemm":
            raise Exception("only gemm problem is defined for numpy")
        if "nb_gpus" in params:
            if params["nb_gpus"] > 0:
                raise Exception("Numpy framework does not work with GPU back-end")

        if isinstance(params["problem"]["size"], int):
            M = N = K = params["problem"]["size"]
        else:
            M, N, K = params["problem"]["size"]
        dtype = np.float32
        a = np.random.random((M, N)).astype(dtype)
        b = np.random.random((N, K)).astype(dtype)
        c = np.random.random((M, K)).astype(dtype)

        nb_epoch = 2

        time_start = timer()
        for _ in range(nb_epoch):
            c = a @ b # + c
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
