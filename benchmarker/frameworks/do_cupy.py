# -*- coding: utf-8 -*-
"""CuPy support.
"""

from timeit import default_timer as timer
import cupy
from .i_gemm import IGEMM


# TODO: this should check if the problem is gemm
class Benchmark(IGEMM):
    def __init__(self, params, remaining_args=None):
        super().__init__(params, remaining_args)

    def run(self):
        if "nb_gpus" in self.params:
            if self.params["nb_gpus"] != 1:
                raise Exception("1 GPU is needed for CuPy")
        M, N, K = self.params["problem"]["size"]
        types = {"FP16": cupy.float16,
                 "FP32": cupy.float32}
        dtype = types[self.params["problem"]["precision"]]
        device = cupy.cuda.Device(device=0)
        device.use()
        # THIS SHOULD GO TO data.py
        # at least we can get the sized from it
        a = cupy.random.random((M, N)).astype(dtype)
        b = cupy.random.random((N, K)).astype(dtype)
        c = cupy.random.random((M, K)).astype(dtype)
        self.data = (a, b, c)
        time_start = timer()
        for _ in range(self.params["nb_epoch"]):
            # TODO: rename net into kernel at some point
            self.net(self.data)
        device.synchronize()
        time_end = timer()
        elapsed_time = (time_end - time_start)
        self.params["time_total"] = elapsed_time
        self.post_process()
