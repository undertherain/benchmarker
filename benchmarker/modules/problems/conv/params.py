import argparse
from ast import literal_eval


def __proc_dims(args, key):
    d = vars(args)
    if isinstance(d[key], str):
        d[key] = literal_eval(d[key])
    if isinstance(d[key], int):
        d[key] = (d[key],) * args.nb_dims

    assert len(d[key]) == args.nb_dims, (
        f"{key} has {len(d[key])} elements, "
        f"while it should have nb_dims = {args.nb_dims}: "
        "dim1, dim2, ..."
    )


def set_extra_params(params, unparsed_args):
    parser = argparse.ArgumentParser(description="Benchmark cuDNN conv kernel")
    parser.add_argument("--cudnn_conv_algo", type=int, default=1)
    parser.add_argument("--nb_dims", type=int, default=2)
    parser.add_argument("--cnt_samples", type=int, default=8)
    parser.add_argument("--input_size", default=50)  # nbDims
    parser.add_argument("--input_channels", type=int, default=3)
    parser.add_argument("--size_kernel", default=3)  # nbDims
    parser.add_argument("--cnt_filters", default=64)
    parser.add_argument("--stride", default=1)  # nbDims
    parser.add_argument("--dilation", default=1)  # nbDims
    parser.add_argument("--padding", default=1)  # nbDims
    args = parser.parse_args(unparsed_args)

    __proc_dims(args, "input_size")
    __proc_dims(args, "size_kernel")
    __proc_dims(args, "stride")
    __proc_dims(args, "dilation")
    __proc_dims(args, "padding")

    params["problem"].update(vars(args))
