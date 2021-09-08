import torch

# from torch_geometric.data import Data


def get_data(params):
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
    sample = {"x": x, "edge_index": edge_index}
    # sample  = (x, edge_index)
    return [sample], [0]
