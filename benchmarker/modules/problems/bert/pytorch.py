from benchmarker.modules.problems.helpers_torch_bert import get_kernel_by_name
from benchmarker.modules.problems.bert_custom import estimate_gflop_per_sample


def get_kernel(params):
    cnt_samples = params["problem"]["size"][0]
    len_seq = params["problem"]["size"][1]
    assert len_seq <= 512, "BERT sequence length must be <= 512 because of saved positional embeddings"
    # TODO: get inner linear size from HF config object
    gflop_per_sample = estimate_gflop_per_sample(
        len_seq=len_seq,
        embed_dim=768,
        lin_dim=3072,
        nb_layers=12,
    )
    gflop_estimated = gflop_per_sample * cnt_samples * params["nb_epoch"]
    params["problem"]["gflop_estimated"] = gflop_estimated
    return get_kernel_by_name(params, "bert-base-uncased")
