"""Helper functions for synthetic data generation"""

import numpy as np


def set_image_size(params, height, width):
    """Set params["problem"]["size"] based on height."""

    if isinstance(params["problem"]["size"], int):
        if params["channels_first"]:
            shape = (3, height, width)
        else:
            shape = (height, width, 3)

    params["problem"]["size"] = (params["problem"]["size"],) + shape


def gen_classification_data(params, height, width=None):
    """Make classification data based on `params["size"]`.
    `param["size"]` has shape (N, H, W, ...) where N is the number of
    samples, and (H, W, ...) is the shape of a single sample. This
    function generates tensor with shape (NB, BS, H, W, ...) i.e. NB *
    BS images/samples and a tensor with shape (NB, BS) of NB * BS
    classes.

    :param params: Dictionary of parameters.

    :return: The images and classes (in batches) as described above.

    """

    if width is None:
        width = height

    set_image_size(params, height, width)

    shape = (params["problem"]["cnt_batches_per_epoch"], params["batch_size"])
    shape = shape + params["problem"]["size"][1:]

    X = np.random.random(shape).astype(np.float32)
    Y = np.random.random(shape[:2]).astype(np.int64)
    return X, Y
