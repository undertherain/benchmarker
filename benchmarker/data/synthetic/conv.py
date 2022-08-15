"""Generate synthetic data for conv layers."""

import re

import numpy as np


def _get_conv_dim(name):
    """Return X in convXd."""
    m = re.match("conv(.)d", name)
    if m:
        groups = m.groups()
        dim = int(groups[0])
        assert 1 <= dim and dim <= 3
        return dim
    else:
        raise ValueError("Wrong input")


def get_data(params):
    """Return np.arrays."""
    problem = params["problem"]
    dim = _get_conv_dim(problem["name"])
    assert len(problem["size"]) == dim + 2

    #shape = (problem["cnt_batches_per_epoch"], params["batch_size"])
    shape = (params["batch_size"],) + problem["size"][1:]
    X = np.random.random(shape).astype(np.float32)
    Y = np.random.random(shape[:2]).astype(np.int64)
    return [{"x": X, "labels": Y}
        for i in range(problem["cnt_batches_per_epoch"])]
