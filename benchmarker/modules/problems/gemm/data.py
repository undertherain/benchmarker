import torch
import numpy as np


def get_data(params):
    M, N, K = params["problem"]["size"]
    flop = (2.0 * M * N * K)
    params["problem"]["flop_estimated"] = flop * params["nb_epoch"]
    if params["framework"] == "torch":
        M, N, K = params["problem"]["size"]
        types = {"FP16": np.float16,
                 "FP32": np.float32,
                 "FP64": np.float64}
        dtype = types[params["problem"]["precision"]]
        a = torch.tensor(np.random.random((M, N)).astype(dtype))
        b = torch.tensor(np.random.random((N, K)).astype(dtype))
        c = torch.tensor(np.random.random((M, K)).astype(dtype))
        return a, b, c
    return ["dummy data"] * 3
