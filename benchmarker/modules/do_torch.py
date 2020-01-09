# -*- coding: utf-8 -*-
"""torch support.
"""

from timeit import default_timer as timer
import torch
import numpy as np
from .i_gemm import IGEMM


class Benchmark(IGEMM):
    def __init__(self, params, remaining_args=None):
        super().__init__(params, remaining_args)

    def run(self):
        M, N, K = self.matrix_size
        dtype = np.float32
        a = torch.tensor(np.random.random((M, N)).astype(dtype))
        b = torch.tensor(np.random.random((N, K)).astype(dtype))
        c = torch.tensor(np.random.random((M, K)).astype(dtype))
        if "nb_gpus" in self.params:
            if self.params["nb_gpus"] > 1:
                raise RuntimeError("Only 1 GPU is supported")
            if self.params["nb_gpus"] == 1:
                torch.cuda.set_device(self.params["gpus"][0])
                a.to("cuda")
                b.to("cuda")
                c.to("cuda")
        nb_epoch = 2
        time_start = timer()
        for _ in range(nb_epoch):
            c = a @ b  # + c
        time_end = timer()
        elapsed_time = (time_end - time_start) / nb_epoch
        self.params["time"] = elapsed_time
        self.params["GFLOP/sec"] = self.params["GFLOP"] / elapsed_time
