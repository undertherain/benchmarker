import torch
import numpy as np


def get_data(params):
    cnt_matrices = params["problem"]["size"][0]
    m = params["problem"]["size"][1]
    n = params["problem"]["size"][2]
    k = params["problem"]["size"][3]
    matr_1 = torch.randn(cnt_matrices, m, n)
    matr_2 = torch.randn(cnt_matrices, n, k)
    Y = np.ones(cnt_matrices, dtype=np.int32)
    return (matr_1, matr_2), Y
