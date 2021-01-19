import argparse


def set_extra_params(params, unparsed_args):
    parser = argparse.ArgumentParser(description='Benchmark kernel')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args(unparsed_args)
    params["batch_size"] = int(args.batch_size)
