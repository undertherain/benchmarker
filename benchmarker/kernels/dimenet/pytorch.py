import torch
# from benchmarker.kernels.helpers_torch import Regression
from torch_geometric.nn import DimeNet


class Wrapper(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def __call__(self, x, y):
        res = self.net(**x)
        loss = torch.nn.functional.mse_loss(res, y)
        return loss


def get_kernel(params):
    net = DimeNet(
        hidden_channels=128,
        out_channels=1,
        num_blocks=6,
        num_bilinear=8,
        num_spherical=7,
        num_radial=6,
        cutoff=5.0,
        envelope_exponent=5,
        num_before_skip=1,
        num_after_skip=2,
        num_output_layers=3)
    return Wrapper(net)
