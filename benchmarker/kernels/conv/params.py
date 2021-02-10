import argparse
from ast import literal_eval


def __make_args_dims_explicit_tuples(args, key):
    # @todo(vatai): custom parser
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


def __set_algo_name(params):
    # Update this from `cudnn.h` using the following commands:
    #
    # grep FWD_ALGO /usr/include/cudnn*8.h | \
    # sed -e 's/.*\(CUDNN_CONVOLUTION[_A-Z]*\).*\([0-9]\+\).*/algo_name[\2] = "\1"/'

    problem = params["problem"]
    idx = problem["cudnn_conv_algo"]

    algo_name = ["" for i in range(9)]
    algo_name[0] = "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM"
    algo_name[1] = "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM"
    algo_name[2] = "CUDNN_CONVOLUTION_FWD_ALGO_GEMM"
    algo_name[3] = "CUDNN_CONVOLUTION_FWD_ALGO_DIRECT"
    algo_name[4] = "CUDNN_CONVOLUTION_FWD_ALGO_FFT"
    algo_name[5] = "CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING"
    algo_name[6] = "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD"
    algo_name[7] = "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED"
    algo_name[8] = "CUDNN_CONVOLUTION_FWD_ALGO_COUNT"

    problem["cudnn_conv_algo_name"] = algo_name[idx]


def set_extra_params(params, unparsed_args):
    parser = argparse.ArgumentParser(description="Benchmark cuDNN conv kernel")
    parser.add_argument("--cudnn_conv_algo", type=int, default=1)
    parser.add_argument("--nb_dims", type=int, default=2)
    parser.add_argument("--nb_epoch", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--input_size", default=50)  # nbDims
    parser.add_argument("--input_channels", type=int, default=3)
    parser.add_argument("--size_kernel", default=3)  # nbDims
    parser.add_argument("--cnt_filters", default=64)
    parser.add_argument("--stride", default=1)  # nbDims
    parser.add_argument("--dilation", default=1)  # nbDims
    parser.add_argument("--padding", default=1)  # nbDims
    args = parser.parse_args(unparsed_args)

    __make_args_dims_explicit_tuples(args, "input_size")
    __make_args_dims_explicit_tuples(args, "size_kernel")
    __make_args_dims_explicit_tuples(args, "stride")
    __make_args_dims_explicit_tuples(args, "dilation")
    __make_args_dims_explicit_tuples(args, "padding")

    params["problem"].update(vars(args))

    __set_algo_name(params)
