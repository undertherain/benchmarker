# import numpy as np
import torch


def get_data(params):
    # if isinstance(params["problem"]["size"], int):
        # params["problem"]["size"] = [params["problem"]["size"]] * 3
    M, N, K = params["problem"]["sample_shape"]
    flop = 2.0 * M * N * K
    params["problem"]["flop_estimated"] = flop * params["nb_epoch"]
    if params["framework"] == "torch":
        types = {"FP16": torch.float16, "FP32": torch.float32, "FP64": torch.float64}
        dtype = types[params["problem"]["precision"]]
        a = torch.rand((M, N), dtype=dtype)
        b = torch.rand((N, K), dtype=dtype)
        c = torch.rand((M, K), dtype=dtype)
        return a, b, c
    return ["dummy data"] * 3
