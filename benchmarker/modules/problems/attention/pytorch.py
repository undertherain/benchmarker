import torch.nn as nn


class Net(nn.MultiheadAttention):
    def forward(self, data):
        super().forward(data, data, data)


def get_kernel(params):
    assert params["mode"] == "inference"
    cnt_samples = params["problem"]["size"][0]
    len_seq = params["problem"]["size"][1]
    embed_dim = params["problem"]["size"][2]
    ops_proj = len_seq * cnt_samples * 8 * embed_dim
    ops_attn = 2 * embed_dim * len_seq * len_seq * cnt_samples
    params["problem"]["ops_estimated"] = (ops_proj + ops_attn) * params["nb_epoch"]

    # expected sizes: cnt_itmes, len_seq, dims
    net = Net(embed_dim=embed_dim,
              num_heads=params["problem"]["cnt_heads"],
              dropout=0.0,
              bias=True,
              add_bias_kv=False,
              add_zero_attn=False,
              kdim=None,
              vdim=None)

    return net
