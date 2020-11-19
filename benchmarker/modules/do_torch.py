# -*- coding: utf-8 -*-
"""torch support.
"""

from timeit import default_timer as timer
import torch
from .i_gemm import IGEMM


def data_to_device(data, device):
    if type(data) == torch.Tensor:
        data = data.to(device)
    else:
        for element in data:
            data_to_device(element, device)


class Benchmark(IGEMM):
    def __init__(self, params, remaining_args=None):
        super().__init__(params, remaining_args)
        self.get_kernel(params, remaining_args)

    def run(self):
        if "nb_gpus" in self.params:
            if self.params["nb_gpus"] > 1:
                raise RuntimeError("Only 1 GPU is supported")
            if self.params["nb_gpus"] == 1:
                device = torch.device("cuda")
                id_gpu = self.params["gpus"][0]
                torch.cuda.set_device(id_gpu)
                data_to_device(self.data, device)
                if self.params["preheat"]:
                    self.net(self.data)
                torch.cuda.synchronize()
        time_start = timer()
        for _ in range(self.params["nb_epoch"]):
            self.net(self.data)
        if self.params["nb_gpus"] == 1:
            torch.cuda.synchronize()
        time_end = timer()
        self.params["time_total"] = (time_end - time_start)
