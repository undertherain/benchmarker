import torch.nn as nn


class Net(nn.MultiheadAttention):
    def forward(self, data):
        super().forward(data, data, data)


def get_kernel(params):
    assert params["mode"] == "inference"
    # expected sizes: cnt_itmes, len_seq, dims
    net = Net(embed_dim=params["problem"]["size"][2],
              num_heads=params["problem"]["cnt_heads"],
              dropout=0.0,
              bias=True,
              add_bias_kv=False,
              add_zero_attn=False,
              kdim=None,
              vdim=None)

    return net
