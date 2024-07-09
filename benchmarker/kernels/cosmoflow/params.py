import argparse
import ast


def set_extra_params(params, unparsed_args):
    parser = argparse.ArgumentParser(description='Benchmark kernel')
    # TODO: be careful about cahnnels positioning
    parser.add_argument("--target_size", default=4)
    parser.add_argument("--dropout", default=0)

    args, unparsed = parser.parse_known_args(unparsed_args)

    params["problem"]["target_size"] = args.target_size
    params["dropout"] = args.dropout

    print(params)
    #exit(0)

    assert unparsed == [], "expected nothing, got" + str(unparsed_args)
