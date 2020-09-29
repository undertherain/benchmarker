import torch.nn as nn


def get_kernel(params):
    assert params["mode"] == "inference"
    Net = nn.GRU(input_size=params["problem"]["size"][2],
                 hidden_size=params["problem"]["cnt_units"],
                 num_layers=params["problem"]["cnt_layers"],
                 bias=True,
                 bidirectional=params["problem"]["bidirectional"])
    return Net
