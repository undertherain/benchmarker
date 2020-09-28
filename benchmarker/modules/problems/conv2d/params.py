import argparse


def set_extra_params(params, unparsed_args):
    print("inside set_extra_params", unparsed_args)
    parser = argparse.ArgumentParser(description='Benchmark conv kernel')
    parser.add_argument('--size_kernel', type=int, default=3)
    parser.add_argument('--cnt_filters', type=int, default=64)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--dilation', type=int, default=1)
    parser.add_argument('--padding', type=int, default=1)
    args = parser.parse_args(unparsed_args)
    params["problem"].update(vars(args))
