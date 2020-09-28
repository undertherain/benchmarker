import torch.nn as nn
# TODO: the CLI interface becoming too clumsy
# TODO: migrade to json configs


def get_kernel(params):
    assert params["mode"] == "inference"
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
