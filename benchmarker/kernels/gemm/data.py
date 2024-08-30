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
        if params["problem"]["precision"] in types:
            dtype = types[params["problem"]["precision"]]
            a = torch.rand((M, N), dtype=dtype)
            b = torch.rand((N, K), dtype=dtype)
            c = torch.rand((M, K), dtype=dtype)
        elif params["problem"]["precision"] == "INT8":
            a = torch.randint(low=-1, high=2, size=(M, N)).to(torch.int8)
            b = torch.randint(low=-1, high=2, size=(N, K)).to(torch.int8)
            c = torch.randint(low=-1, high=2, size=(M, K)).to(torch.int8)
        else:
            raise ValueError(f"can not recognize {params["problem"]["precision"] }")
        return a, b, c
    return ["dummy data"] * 3
