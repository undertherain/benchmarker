import argparse


def set_extra_params(params, unparsed_args):
    parser = argparse.ArgumentParser(description="Benchmark cuDNN conv kernel")
    parser.add_argument("--cudnn_conv_algo", type=int, default=2)
    parser.add_argument("--nb_dims", type=int, default=2)
    parser.add_argument("--size_kernel", default=3)
    parser.add_argument("--cnt_filters", default=64)
    parser.add_argument("--stride", default=1)
    parser.add_argument("--dilation", default=1)
    parser.add_argument("--padding", default=1)
    args = parser.parse_args(unparsed_args)
    print(f"args.nb_dims = {args.nb_dims}")
    params["problem"].update(vars(args))
