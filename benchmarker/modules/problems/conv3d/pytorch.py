import torch.nn as nn


def get_kernel(params, unparsed_args=None):
    assert params["mode"] == "inference"
    print(params["problem"])
    problem_params = params["problem"]
    Net = nn.Conv3d(
        in_channels=params["problem"]["size"][1],
        out_channels=problem_params["cnt_filters"],
        kernel_size=problem_params["size_kernel"],
        stride=problem_params["stride"],
        padding=problem_params["padding"],
        dilation=problem_params["dilation"],
        groups=1,
        bias=True,
        padding_mode="zeros",
    )
    return Net
