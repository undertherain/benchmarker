import argparse
import torch.nn as nn

# TODO: the CLI interface becoming too clumsy
# TODO: migrade to json configs


def get_kernel(params, unparsed_args=None):
    assert params["mode"] == "inference"
    parser = argparse.ArgumentParser(description='Benchmark conv1d kernel')
    parser.add_argument('--size_kernel', type=int, default=3)
    parser.add_argument('--cnt_filters', type=int, default=64)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--dilation', type=int, default=1)
    parser.add_argument('--padding', type=int, default=0)
    args = parser.parse_args(unparsed_args)
    params["problem"].update(vars(args))
    print(params["problem"])
    Net = nn.Conv1d(in_channels=params["problem"]["size"][1],
                    out_channels=args.cnt_filters,
                    kernel_size=args.size_kernel,
                    stride=args.stride,
                    padding=args.padding,
                    dilation=args.dilation,
                    groups=1,
                    bias=True,
                    padding_mode='zeros')
    return Net
