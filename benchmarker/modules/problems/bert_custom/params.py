import argparse
import ast


def set_extra_params(params, unparsed_args):
    parser = argparse.ArgumentParser(description='Benchmark kernel')
    parser.add_argument("--target_size", default=4)
    parser.add_argument("--dropout", default=0)

    args = parser.args(unparsed_args)
    # TODO(Alex): this should be in problem sub-dict
    params["input_shape"] = ast.literal_eval(args.input_shape)
    params["target_size"] = args.target_size
    params["dropout"] = args.dropout
