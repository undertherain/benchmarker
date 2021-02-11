import torch


def get_data(params):
    batch_size = params["batch_size"]
    M, N, K = params["problem"]["size"]
    flop = (2.0 * M * N * K)
    params["problem"]["flop_estimated"] = flop * params["nb_epoch"] * batch_size
    m = params["problem"]["size"][0]
    n = params["problem"]["size"][1]
    k = params["problem"]["size"][2]
    matr_1 = torch.randn(batch_size, m, n)
    matr_2 = torch.randn(batch_size, n, k)
    return matr_1, matr_2
