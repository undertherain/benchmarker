import torch.nn as nn
from .params import set_extra_params
# TODO: the CLI interface becoming too clumsy
# TODO: migrade to json configs


def get_kernel(params, unparsed_args=None):
    assert params["mode"] == "inference"
    args = set_extra_params(unparsed_args)
    params["problem"].update(vars(args))
    print(params["problem"])
    Net = nn.Conv2d(in_channels=params["problem"]["size"][1],
                    out_channels=args.cnt_filters,
                    kernel_size=args.size_kernel,
                    stride=args.stride,
                    padding=args.padding,
                    dilation=args.dilation,
                    groups=1,
                    bias=True,
                    padding_mode='zeros')
    return Net
