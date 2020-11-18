import torch


class Net:
    def __call__(data):
        x, y = data
        result = torch.bmm(x, y)
        return result.size()


def get_kernel(params):
    assert params["mode"] == "inference"
    return Net
