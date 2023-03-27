"""Generate synthetic data for 224x224 images segmentation problem"""
import numpy as np

from .helpers import set_image_size


def get_data(params):
    # set_image_size(params, 224, 224)

    if isinstance(params["problem"]["size"], int):
        params["problem"]["size"] = (params["problem"]["size"], 3, 224, 224)

    shape = (params["batch_size"],) + params["problem"]["size"][1:]
    # print("shape", shape)
    # TODO: num classes in place of num channels
    images = np.random.random(shape).astype(np.float32)
    # shape = (params["problem"]["cnt_batches_per_epoch"], params["batch_size"])
    # shape = shape + params["problem"]["size"][2:]
    masks = np.ones(shape, dtype=np.int64)
    # Y = np.random.randint(0, num_cls, shape[:1])
    name_key = "x"
    return [{name_key: images, "labels": masks} for i in range(params["problem"]["cnt_batches_per_epoch"],)]
    # return images, masks
