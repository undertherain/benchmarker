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
        self.a, self.b, self.c = self.load_data()

    def run(self):
        if "nb_gpus" in self.params:
            if self.params["nb_gpus"] > 1:
                raise RuntimeError("Only 1 GPU is supported")
            if self.params["nb_gpus"] == 1:
                device = torch.device("cuda")
                id_gpu = self.params["gpus"][0]
                torch.cuda.set_device(id_gpu)
                self.a = self.a.to(device)
                self.b = self.b.to(device)
                self.c = self.c.to(device)
                if self.params["preheat"]:
                    self.c = self.a @ self.b  # this is preheat
                torch.cuda.synchronize()
        time_start = timer()
        for _ in range(self.params["nb_epoch"]):
            self.c = self.a @ self.b  # + c
        if self.params["nb_gpus"] == 1:
            torch.cuda.synchronize()
        time_end = timer()
        self.params["time_total"] = (time_end - time_start)
