import argparse
import torch.nn as nn

# TODO: move this to params
cnt_channels = 3

# TODO: the CLI interface becoming too clumsy
# TODO: migrade to json configs


def get_kernel(params, unparsed_args=None):
    assert params["mode"] == "inference"
    parser = argparse.ArgumentParser(description='Benchmark conv kernel')
    parser.add_argument('--size_kernel', type=int, default=3)
    parser.add_argument('--cnt_filters', type=int, default=64)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--dilation', type=int, default=1)
    args = parser.parse_args(unparsed_args)
    params["problem"].update(vars(args))
    print(params["problem"])
    Net = nn.Conv2d(in_channels=cnt_channels,
                    out_channels=args.cnt_filters,
                    kernel_size=args.size_kernel,
                    stride=args.stride,
                    padding=1,
                    dilation=args.dilation,
                    groups=1,
                    bias=True,
                    padding_mode='zeros')
    return Net
