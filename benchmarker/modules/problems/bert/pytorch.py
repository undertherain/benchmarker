from benchmarker.modules.problems.attention.pytorch import \
    estimate_attention_gflop_per_sample
from benchmarker.modules.problems.helpers_torch_bert import get_kernel_by_name


def estimate_gflop_per_sample(len_seq, embed_dim, lin_dim, nb_layers):
    attention_gflop = estimate_attention_gflop_per_sample(len_seq, embed_dim)
    mid_linear_gflop = 2 * embed_dim * lin_dim / (10 ** 9)
    top_linear_gflop = 2 * embed_dim * lin_dim / (10 ** 9)
    layer_gflop = attention_gflop + mid_linear_gflop + top_linear_gflop
    # TODO: add layer norm
    # TODO: get inner linear size from HF config object
    return nb_layers * layer_gflop


def get_kernel(params):
    cnt_samples = params["problem"]["size"][0]
    gflop_per_sample = estimate_gflop_per_sample(
        len_seq=params["problem"]["size"][1],
        embed_dim=768,
        lin_dim=3072,
        nb_layers=12,
    )
    gflop_estimated = gflop_per_sample * cnt_samples * params["nb_epoch"]
    params["problem"]["gflop_estimated"] = gflop_estimated
    return get_kernel_by_name(params, "bert-base-uncased")
