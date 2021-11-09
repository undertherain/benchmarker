# from benchmarker.kernels.helpers_torch_bert import get_kernel_by_name
import torch
from benchmarker.kernels.bert_custom import estimate_gflop_per_sample
from transformers import BertConfig, BertForMaskedLM


class Wrapper(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, *input, **kwargs):
        res = self.net(*input, **kwargs)
        return res.loss


def get_kernel(params):
    # cnt_samples = params["problem"]["size"][0]
    len_seq = params["problem"]["size"][1]
    assert len_seq <= 512, "BERT sequence length must be <= 512 because of saved positional embeddings"
    # TODO: get inner linear size from HF config object
    # gflop_per_sample = estimate_gflop_per_sample(
    #     len_seq=len_seq,
    #     embed_dim=768,
    #     lin_dim=3072,
    #     nb_layers=12,
    # )
    #gflop_estimated = gflop_per_sample * cnt_samples * params["nb_epoch"]
    # params["problem"]["gflop_estimated"] = gflop_estimated
    config = BertConfig.from_pretrained("bert-base-uncased")
    net = BertForMaskedLM._from_config(config)
    return Wrapper(net)
