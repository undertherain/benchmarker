import torch.nn as nn


class Net(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.net = nn.LSTM(input_size=params["problem"]["size"][2],
                           hidden_size=params["problem"]["cnt_units"],
                           num_layers=params["problem"]["cnt_layers"],
                           bias=True,
                           bidirectional=params["problem"]["bidirectional"])

    def __call__(self, x, labels):
        return self.net(x)


def get_kernel(params):
    assert params["mode"] == "inference"
    return Net(params)
