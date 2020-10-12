import torch.nn as nn


class Net(nn.MultiheadAttention):
    def forward(self, data):
        super().forward(data, data, data)


def get_kernel(params):
    assert params["mode"] == "inference"
    cnt_samples = params["problem"]["size"][0]
    len_seq = params["problem"]["size"][1]
    embed_dim = params["problem"]["size"][2]
    cnt_projections = 4
    ops_proj = 2 * cnt_samples * len_seq * embed_dim * cnt_projections
    ops_Q_K = 2 * cnt_samples * len_seq * len_seq * embed_dim
    ops_Q_Kt_V = ops_Q_K
    params["problem"]["flop_estimated"] = (ops_proj + ops_Q_K + ops_Q_Kt_V) * params["nb_epoch"]

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
