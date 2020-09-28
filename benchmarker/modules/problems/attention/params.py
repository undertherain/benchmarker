import argparse


def set_extra_params(params, unparsed_args):
    parser = argparse.ArgumentParser(description='Benchmark kernel')
    parser.add_argument('--cnt_heads', type=int, default=8)
    args = parser.parse_args(unparsed_args)
    params["problem"].update(vars(args))
