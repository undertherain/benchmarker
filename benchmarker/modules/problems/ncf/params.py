import argparse


def set_extra_params(params, unparsed_args):
    parser = argparse.ArgumentParser(description='Benchmark kernel')
    parser.add_argument("--nb_users", type=int, default=128, help="number of users")
    parser.add_argument("--nb_items", type=int, default=128, help="number of items")
    parser.add_argument(
        "--factors", type=int, default=8, help="number of predictive factors"
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=[64, 32, 16, 8],
        help="size of hidden layers for MLP",
    )
    args = parser.parse_args(unparsed_args)
    params["problem"].update(vars(args))
