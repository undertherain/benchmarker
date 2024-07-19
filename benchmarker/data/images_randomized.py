import numpy as np


# TODO: this repeats code from .synthetic
def gen_data(params):
    """generates sinthetic dataset"""
    # input size -> cnt_sequences, len_sequence, cnt_dimensions
    # transform into (seq_len, batch, cnt_dimentions) x cnt_batches
    params["problem"]["len_sequence"] = params["problem"]["sample_shape"][0]
    cnt_batches = params["problem"]["cnt_batches_per_epoch"]
    shape = (
        params["problem"]["len_sequence"],
        params["batch_size"],
        params["problem"]["sample_shape"][1],
    )
    data = [{"x": np.random.random(shape).astype(np.float32),
             "labels": np.ones((params["batch_size"]))} for i in range(cnt_batches)]
    return data
