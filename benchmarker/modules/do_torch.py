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
        if "nb_gpus" in self.params:
            if self.params["nb_gpus"] > 0:
                raise NotImplementedError("wip on GPU execution")
        M, N, K = self.matrix_size
        dtype = np.float32
        a = torch.tensor(np.random.random((M, N)).astype(dtype))
        b = torch.tensor(np.random.random((N, K)).astype(dtype))
        c = torch.tensor(np.random.random((M, K)).astype(dtype))
        nb_epoch = 2
        time_start = timer()
        for _ in range(nb_epoch):
            c = a @ b # + c
        time_end = timer()
        elapsed_time = (time_end - time_start) / nb_epoch
        self.params["time"] = elapsed_time
        self.params["GFLOP/sec"] = self.params["GFLOP"] / elapsed_time
