# -*- coding: utf-8 -*-
"""Keras support.

Used for Tensofrlow and Theano
"""

from timeit import default_timer as timer
import importlib
from benchmarker.util.data import to_categorical


def run(params, data):
    # if params["channels_first"]:
    #     keras.backend.set_image_data_format("channels_first")
    # else:
    #     keras.backend.set_image_data_format("channels_last")

    x_train, y_train = data

    y_train = to_categorical(y_train, num_classes=1000)

    mod = importlib.import_module("benchmarker.kernels." +
                                  params["problem"]["name"] + ".keras")
    get_model = getattr(mod, 'get_model')

    if len(y_train.shape) > 1:
        cnt_classes = y_train.shape[1]
    else:
        cnt_classes = 1
    params["cnt_classes"] = cnt_classes
    model = get_model(params)
    if params["mode"] != "training":
        raise NotADirectoryError("only training is implemented for TF")
    print("preheat")
    model.fit(x_train, y_train, batch_size=params["batch_size"], epochs=1)
    nb_epoch = 3
    print("train")
    start = timer()
    model.fit(x_train, y_train, batch_size=params["batch_size"], epochs=nb_epoch, verbose=1)
    end = timer()
    params["time"] = (end - start) / nb_epoch
    if params["framework"] == "theano":
        import theano
        version_backend = theano.__version__
    else:
        import tensorflow as tf
        version_backend = tf.__version__
    # TODO: make this a nested dict
    # params["framework_full"] = "Keras-" + keras.__version__ + "/" + keras.backend.backend() + "_" + version_backend
    params["framework_full"] = "TensorFlow-" + version_backend
    return params
