# -*- coding: utf-8 -*-
"""CuPy support.
"""

from timeit import default_timer as timer
import cupy
from .i_gemm import IGEMM


class Benchmark(IGEMM):
    def __init__(self, params, remaining_args=None):
        super().__init__(params, remaining_args)

    def run(self):
        if "nb_gpus" in self.params:
            if self.params["nb_gpus"] != 1:
                raise Exception("1 GPU is needed for CuPy")
        M, N, K = self.matrix_size
        dtype = cupy.float32
        device = cupy.cuda.Device(device=0)
        device.use()
        a = cupy.random.random((M, N)).astype(dtype)
        b = cupy.random.random((N, K)).astype(dtype)
        c = cupy.random.random((M, K)).astype(dtype)
        nb_epoch = 4
        time_start = timer()
        for _ in range(nb_epoch):
            c = a @ b # + c
        device.synchronize()
        time_end = timer()
        elapsed_time = (time_end - time_start) / nb_epoch
        self.params["time"] = elapsed_time
        self.params["GFLOP/sec"] = self.params["GFLOP"] / elapsed_time
