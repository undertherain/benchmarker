from torch_geometric.nn import DimeNet


def get_kernel(params):
    net = DimeNet(
        hidden_channels=128,
        out_channels=128,
        num_blocks=3,
        num_bilinear=8,
        num_spherical=7,
        num_radial=6)
    return net
