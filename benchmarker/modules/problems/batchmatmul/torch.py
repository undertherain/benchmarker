import torch


class Net:
    def __call__(self, data):
        x, y = data
        # print(y)
        result = torch.bmm(x, y)
        return result.size()


def get_kernel(params):
    return Net()
