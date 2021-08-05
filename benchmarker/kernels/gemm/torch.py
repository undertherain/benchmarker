import torch


class Kernel:
    def __call__(self, data):
        x, y, c = tuple(map(torch.tensor, data))
        c = x @ y  # + c
        return c


def get_kernel(params):
    return Kernel()
