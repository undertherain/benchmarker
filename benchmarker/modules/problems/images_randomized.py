import numpy as np


# TODO: get back generation of somewhat patterned images
def gen_images(params):
    """Generates synthetic dataset."""

    shape = (params["problem"]["cnt_batches_per_epoch"], params["batch_size"])
    shape = shape + params["problem"]["size"][1:]

    # TODO(vatai): Not sure if these asserts are neede.
    if (params["problem"]["name"]) == "conv1d":
        # (cnt_batches, batch, channels, x)
        assert len(params["problem"]["size"]) == 3
    elif (params["problem"]["name"]) == "conv2d":
        # (cnt_batches, batch, channels, x, h,w)
        assert len(params["problem"]["size"]) == 4
    elif (params["problem"]["name"]) == "conv3d":
        # (cnt_batches, batch, channels, x, h,w)
        assert len(params["problem"]["size"]) == 5

    X = np.random.random(shape).astype(np.float32)
    Y = np.random.random(shape[:2]).astype(np.int64)
    return X, Y

def gen_data(params):
    """generates sinthetic dataset"""
    #input size -> cnt_sequences, len_sequence, cnt_dimensions
    #transform into (seq_len, batch, cnt_dimentions) x cnt_batches
    assert params["problem"]["size"][0] % params["batch_size"] == 0
    params["problem"]["len_sequence"] = params["problem"]["size"][1]
    cnt_batches = params["problem"]["size"][0] // params["batch_size"]
    shape = (cnt_batches,
             params["problem"]["len_sequence"],
             params["batch_size"],
             params["problem"]["size"][2])
    X = np.random.random(shape).astype(np.float32)
    Y = np.ones((cnt_batches))
    return X, Y

def test():
    params = {
        "batch_size": 32,
        "problem": {
            "name": "conv2d",
            "cnt_batches_per_epoch": 16,
            "size": [10, 20, 30, 40],
        },
    }
    X, Y = gen_images(params)

    assert X.shape == (16, 32, 20, 30, 40)
    assert Y.shape == (16, 32)


if __name__ == "__main__":
    test()
