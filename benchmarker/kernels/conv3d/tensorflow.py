from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.optimizers import SGD


def get_kernel(params):
    assert params["mode"] == "inference"
    if params["problem"]["padding"] == 1:
        params["problem"]["padding"] = "valid"
    conv1d = Conv3D(
        filters=params["problem"]["cnt_filters"],
        kernel_size=params["problem"]["size_kernel"],
        strides=params["problem"]["stride"],
        dilation_rate=params["problem"]["dilation"],
        padding=params["problem"]["padding"],
    )
    model = Sequential([conv1d])
    model.compile(optimizer=SGD())
    return model
