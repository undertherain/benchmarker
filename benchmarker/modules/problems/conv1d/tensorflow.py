from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.optimizers import SGD


def get_kernel(params, unparsed_args):
    assert params["mode"] == "inference"
    conv1d = Conv1D(
        filters=params["problem"]["cnt_filters"],
        kernel_size=params["problem"]["size_kernel"],
        strides=params["problem"]["stride"],
        dilation_rate=params["problem"]["dilation"],
        padding=params["problem"]["padding"],
    )
    model = Sequential([conv1d])
    model.compile(optimizer=SGD())
    return model
