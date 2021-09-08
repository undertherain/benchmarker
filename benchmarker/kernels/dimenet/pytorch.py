import torch
# from benchmarker.kernels.helpers_torch import Regression
from torch_geometric.nn import DimeNet


class Wrapper(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def __call__(self, x, y):
        res = self.net(**x)
        loss = torch.nn.functional.mse_loss(res.sum(), y)
        return loss


def get_kernel(params):
    net = DimeNet(
        hidden_channels=16,
        out_channels=16,
        num_blocks=3,
        num_bilinear=2,
        num_spherical=4,
        num_radial=4)
    return Wrapper(net)
