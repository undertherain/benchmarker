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
        types = {"mixed": np.float32,
                 "FP32": np.float32,
                 "FP16": np.float16}
        dtype = types[self.params["problem"]["precision"]]
        a = torch.tensor(np.random.random((M, N)).astype(dtype))
        b = torch.tensor(np.random.random((N, K)).astype(dtype))
        c = torch.tensor(np.random.random((M, K)).astype(dtype))
        if "nb_gpus" in self.params:
            if self.params["nb_gpus"] > 1:
                raise RuntimeError("Only 1 GPU is supported")
            if self.params["nb_gpus"] == 1:
                device = torch.device("cuda")
                id_gpu = self.params["gpus"][0]
                torch.cuda.set_device(id_gpu)
                a = a.to(device)
                b = b.to(device)
                c = c.to(device)
                # pre-heat
                c = a @ b
                torch.cuda.synchronize()
        nb_epoch = 2
        time_start = timer()
        for _ in range(nb_epoch):
            c = a @ b  # + c
        if self.params["nb_gpus"] == 1:
            torch.cuda.synchronize()
        time_end = timer()
        elapsed_time = (time_end - time_start) / nb_epoch
        self.params["time"] = elapsed_time
        self.params["GFLOP/sec"] = self.params["GFLOP"] / elapsed_time
