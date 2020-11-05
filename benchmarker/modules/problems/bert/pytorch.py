from benchmarker.modules.problems.helpers_torch_bert import get_kernel_by_name
from benchmarker.modules.problems.attention.pytorch import estimate_attention_gflop_per_sample


def get_kernel(params):
    gflop_per_sample = estimate_attention_gflop_per_sample(len_seq, embed_dim)
    # add dense
    # * num layers
    params["problem"]["gflop_estimated"] = gflop_per_sample * cnt_samples * params["nb_epoch"]
    return get_kernel_by_name(params, "bert-base-uncased")
