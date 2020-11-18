import torch


class Net:
    def __call__(data):
        x, y = data
        result = x @ y
        return result


def get_kernel(params):
    return Net
