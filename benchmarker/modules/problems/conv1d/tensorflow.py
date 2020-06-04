import argparse

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.optimizers import SGD


def get_kernel(params, unparsed_args):
    assert params["mode"] == "inference"
    parser = argparse.ArgumentParser(description="Benchmark conv1d kernel")
    parser.add_argument("--size_kernel", type=int, default=3)
    parser.add_argument("--cnt_filters", type=int, default=64)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--dilation", type=int, default=1)
    parser.add_argument("--padding", type=str, default="valid")
    args = parser.parse_args(unparsed_args)
    params["problem"].update(vars(args))
    print(params["problem"])

    conv1d = Conv1D(
        filters=args.cnt_filters,
        kernel_size=args.size_kernel,
        strides=args.stride,
        dilation_rate=args.dilation,
        padding=args.padding,
    )
    model = Sequential([conv1d])
    model.compile(optimizer=SGD())
    return model
