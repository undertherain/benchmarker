"""Generate synthetic data for 224x224 images segmentation problem"""
import numpy as np
from .helpers import set_image_size


def get_data(params):
    """Generate synthetic 224x224 images. Set `params["size"]`
    appropriately. Import this function in the `data.py` of the
    problem, so it can be called by `INeuralNet`.

    """
    set_image_size(params, 224, 224)
    shape = (params["problem"]["cnt_batches_per_epoch"], params["batch_size"])
    shape = shape + params["problem"]["size"][1:]

    # TODO: num classes in place of num channels
    images = np.random.random(shape).astype(np.float32)
    shape = (params["problem"]["cnt_batches_per_epoch"], params["batch_size"])
    shape = shape + params["problem"]["size"][2:]
    masks = np.ones(shape, dtype=np.int64)
    return images, masks
