# -*- coding: utf-8 -*-
"""torch support.
"""

import argparse
from timeit import default_timer as timer

import torch

from .i_gemm import IGEMM


def data_to_device(data, device):
    if type(data) == torch.Tensor:
        return data.to(device)
    else:
        return tuple(data_to_device(element, device) for element in data)


class Benchmark(IGEMM):
    def __init__(self, params, extra_args=None):
        # TODO: reuse this with do_pytorch
        parser = argparse.ArgumentParser(description="pytorch extra args")
        parser.add_argument("--backend", default="native")
        args, remaining_args = parser.parse_known_args(extra_args)
        super().__init__(params, remaining_args)
        self.data = tuple(map(torch.tensor, self.data))
        self.params["backend"] = args.backend
        # self.get_kernel(params, remaining_args)
        if self.params["backend"] == "DNNL":
            torch.backends.mkldnn.enabled = True
            self.data = (self.data[0].to_mkldnn(), self.data[1].to_mkldnn())
        else:
            if self.params["backend"] == "native":
                torch.backends.mkldnn.enabled = False
            else:
                raise RuntimeError("Unknown backend")

    def run(self):
        if "nb_gpus" in self.params:
            if self.params["nb_gpus"] > 1:
                raise RuntimeError("Only 1 GPU is supported")
            if self.params["nb_gpus"] == 1:
                device = torch.device("cuda")
                id_gpu = self.params["gpus"][0]
                torch.cuda.set_device(id_gpu)
                self.data = data_to_device(self.data, device)
                # self.net.to(device)
                if self.params["preheat"]:
                    self.net(self.data)
                torch.cuda.synchronize()
        # preheat
        self.net(self.data)
        time_start = timer()
        for _ in range(self.params["nb_epoch"]):
            self.net(self.data)
        if self.params["nb_gpus"] == 1:
            torch.cuda.synchronize()
        time_end = timer()
        self.params["time_total"] = time_end - time_start
