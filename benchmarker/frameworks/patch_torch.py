# from time import sleep

from functools import reduce
from operator import mul

import torch

orig_bmm = torch.bmm
orig_matmul = torch.matmul
# orig_addmm = torch.addbmm
orig_linear = torch.nn.functional.linear


def wrap_bmm(input, mat2, *args, out=None):
    # assert out is None
    # assert not args
    # TODO: call original
    # print("CALLING PATCHED BMM")
    print("BMM", input.shape, mat2.shape)
    return orig_bmm(input, mat2)


def print_mnk(shape_1, shape_2):
    # (m*k) x (k*n) = (m*n)
    assert shape_1[-1] == shape_2[-2]
    bs1 = reduce(mul, [1] + list(shape_1[:-2]))
    bs2 = reduce(mul, [1] + list(shape_2[:-2]))
    bs = max(bs1, bs2)
    print(f"M={shape_1[-2]}, N={shape_2[-1]}, K={shape_1[-1]} bs={bs}",)


def wrap_matmul(input, other, *args, out=None):
    # cache = dict()
    # assert out is None
    # assert not args
    # TODO: call original
    # print("CALLING PATCHED MATMUL")
    # input.cuda()
    # other.cuda()
    # return orig_matmul(input, other).cpu()
    # raise RuntimeError("oi")
    print("matmul", input.shape, other.shape)
    print_mnk(input.shape, other.shape)
    return orig_matmul(input, other, *args)
    #shape = input.shape[: -2] + (input.shape[-2],) + (other.shape[-1],)
    #return torch.ones(shape)
    # if shape not in cache:
    #     cache[shape] = torch.ones(shape)
    # print(input.shape, other.shape, shape)
    # sleep(0.1)
    # return cache[shape]


def wrap_linear(input, weight, bias=...):
    print("linear", input.shape, weight.T.shape)
    print_mnk(input.shape, weight.T.shape)
    return orig_linear(input, weight, bias)
    return torch.ones((input.shape[0], input.shape[1], weight.shape[0]))
    i = input.cuda()
    w = weight.cuda()
    b = bias.cuda()
    result = orig_linear(i, w, b)
    return result.cpu()
    # print(input.shape, weight.shape, result.shape)


def patch_bmm():
    torch.bmm = wrap_bmm
    torch.matmul = wrap_matmul
    torch.nn.functional.linear = wrap_linear
    # pass
