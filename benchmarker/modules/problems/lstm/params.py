import argparse


def set_extra_params(params, unparsed_args):
    print("inside set_extra_params", unparsed_args)
    parser = argparse.ArgumentParser(description='Benchmark kernel')
    parser.add_argument('--cnt_units', type=int, default=512)
    parser.add_argument('--cnt_layers', type=int, default=1)
    parser.add_argument('--bidirectional', type=bool, default=False)
    args = parser.parse_args(unparsed_args)
    params["problem"].update(vars(args))
