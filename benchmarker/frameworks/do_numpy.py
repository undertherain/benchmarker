# -*- coding: utf-8 -*-
"""NumPy support.
"""

from timeit import default_timer as timer
import numpy as np
from .i_gemm import IGEMM


class Benchmark(IGEMM):
    def __init__(self, params, remaining_args=None):
        super().__init__(params, remaining_args)

    def run(self):
        if "nb_gpus" in self.params:
            if self.params["nb_gpus"] > 0:
                raise Exception("Numpy framework does not work with GPU back-end")
        M, N, K = self.params["problem"]["size"]
        types = {"FP32": np.float32,
                 "FP16": np.float16}
        dtype = types[self.params["problem"]["precision"]]
        a = np.random.random((M, N)).astype(dtype)
        b = np.random.random((N, K)).astype(dtype)
        # c = np.random.random((M, K)).astype(dtype)
        time_start = timer()
        for _ in range(self.params["nb_epoch"]):
            c = a @ b  # + c
        time_end = timer()
        elapsed_time = (time_end - time_start)
        self.params["time_total"] = elapsed_time
        self.post_process()
