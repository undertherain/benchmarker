import numpy as np


# TODO: get back generation of somewhat patterned images
def gen_images(params):
    #"""generates sinthetic dataset"""

    if (params["problem"]["name"]) == 'conv1d':
        # (cnt_batches, batch, channels, x)
        assert len(params["problem"]["size"]) == 3

        shape = (params["problem"]["cnt_batches_per_epoch"],
                 params["batch_size"],
                 params["problem"]["size"][1],
                 params["problem"]["size"][2])

    elif (params["problem"]["name"]) == 'conv2d':
      # (cnt_batches, batch, channels, x, h,w)
        assert len(params["problem"]["size"]) == 4

        shape = (params["problem"]["cnt_batches_per_epoch"],
                 params["batch_size"],
                 params["problem"]["size"][1],
                 params["problem"]["size"][2],
                 params["problem"]["size"][3])

    elif (params["problem"]["name"]) == 'conv3d':
        # (cnt_batches, batch, channels, x, h,w)
        assert len(params["problem"]["size"]) == 5

        shape = (params["problem"]["cnt_batches_per_epoch"],
                 params["batch_size"],
                 params["problem"]["size"][1],
                 params["problem"]["size"][2],
                 params["problem"]["size"][3],
                 params["problem"]["size"][4])

    X = np.random.random(shape).astype(np.float32)
    Y = np.random.random(shape[:2]).astype(np.int64)
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
