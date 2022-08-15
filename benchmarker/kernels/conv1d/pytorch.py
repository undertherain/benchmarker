import torch.nn as nn
from benchmarker.kernels.helpers_torch import InferenceOnly


def get_kernel(params):
    assert params["mode"] == "inference"
    print(params["problem"])
    problem_params = params["problem"]
    net = nn.Conv1d(in_channels=params["problem"]["size"][1],
                    out_channels=problem_params["cnt_filters"],
                    kernel_size=problem_params["size_kernel"],
                    stride=problem_params["stride"],
                    padding=problem_params["padding"],
                    dilation=problem_params["dilation"],
                    groups=1,
                    bias=True,
                    padding_mode='zeros')
    return InferenceOnly(params["mode"], net)
