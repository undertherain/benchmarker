import torch.nn as nn
from .params import set_extra_params
# TODO: the CLI interface becoming too clumsy
# TODO: migrade to json configs


def get_kernel(params, unparsed_args=None):
    assert params["mode"] == "inference"
    set_extra_params(params, unparsed_args)
    print(params["problem"])
    problem_params = params["problem"]
    Net = nn.Conv2d(in_channels=problem_params["size"][1],
                    out_channels=problem_params["cnt_filters"],
                    kernel_size=problem_params["size_kernel"],
                    stride=problem_params["stride"],
                    padding=problem_params["padding"],
                    dilation=problem_params["dilation"],
                    groups=1,
                    bias=True,
                    padding_mode='zeros')
    return Net
