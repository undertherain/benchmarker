import argparse
import ast


def proc_params(params, unparsed_args):
    """Process args for CosmoFlow in (D1, D2, D3, Channels) tuple."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_shape", default="128, 128, 128, 4")
    parser.add_argument("--target_size", default=4)
    parser.add_argument("--dropout", default=0)

    args, unparsed = parser.parse_known_args(unparsed_args)

    params["input_shape"] = ast.literal_eval(args.input_shape)
    params["target_size"] = args.target_size
    params["dropout"] = args.dropout

    assert unparsed == []
