import argparse


def set_extra_params(params, unparsed_args):
    parser = argparse.ArgumentParser(description='Benchmark kernel')
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--cnt_atoms_in_sample", type=int, default=6)

    args = parser.parse_args(unparsed_args)
    params["problem"].update(vars(args))
