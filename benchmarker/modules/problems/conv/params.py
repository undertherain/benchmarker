import argparse
from ast import literal_eval


def set_extra_params(params, unparsed_args):
    parser = argparse.ArgumentParser(description="Benchmark cuDNN conv kernel")
    parser.add_argument("--cudnn_conv_algo", type=int, default=2)
    parser.add_argument("--nb_dims", type=int, default=2)
    parser.add_argument("--cnt_samples", type=int, default=8)
    parser.add_argument("--input_size", default=(100, 100))
    parser.add_argument("--input_channels", type=int, default=3)
    parser.add_argument("--size_kernel", default=3)
    parser.add_argument("--cnt_filters", default=64)
    parser.add_argument("--stride", default=1)
    parser.add_argument("--dilation", default=1)
    parser.add_argument("--padding", default=1)
    args = parser.parse_args(unparsed_args)

    if isinstance(args.input_size, str):
        args.input_size = literal_eval(args.input_size)
    assert len(args.input_size) == args.nb_dims, (
        f"input_size has {len(args.input_size)} number of elements, "
        f"while it should be nb_dims = {args.nb_dims}: "
        "dim1, dim2, ..."
    )
    params["problem"].update(vars(args))
