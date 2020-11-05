import torch.nn as nn


class Net(nn.MultiheadAttention):
    def forward(self, data):
        super().forward(data, data, data)


def estimate_attention_gflop_per_sample(len_seq, embed_dim):
    cnt_projections = 4
    ops_proj = 2 * len_seq * embed_dim * embed_dim * cnt_projections
    ops_Q_K = 2 * len_seq * len_seq * embed_dim
    ops_Q_Kt_V = ops_Q_K
    ops_estimated = ((ops_proj + ops_Q_K + ops_Q_Kt_V)) / (10 ** 9)
    return ops_estimated


def get_kernel(params):
    assert params["mode"] == "inference"
    cnt_samples = params["problem"]["size"][0]
    len_seq = params["problem"]["size"][1]
    embed_dim = params["problem"]["size"][2]
    gflop_per_sample = estimate_attention_gflop_per_sample(len_seq, embed_dim)
    params["problem"]["gflop_estimated"] = gflop_per_sample * cnt_samples * params["nb_epoch"]
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
