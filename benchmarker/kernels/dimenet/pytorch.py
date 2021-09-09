from benchmarker.kernels.helpers_torch import Regression
from torch_geometric.nn import DimeNet


def get_kernel(params):
    # TODO: make these parameters
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
    return Regression(params["mode"], net)
