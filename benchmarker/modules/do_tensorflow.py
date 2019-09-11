# -*- coding: utf-8 -*-
"""TensorFlow support.
"""

import os
from timeit import default_timer as timer
from .i_neural_net import INeuralNet
from benchmarker.util.data import to_categorical
import tensorflow as tf


class DoTensorflow(INeuralNet):
    """docstring for ClassName"""

    def __init__(self, params):
        super().__init__(params)
        self.params["channels_first"] = False

    def run_internal(self):
        # todo set image format
        data = self.load_data()

        os.environ["KERAS_BACKEND"] = "tensorflow"
        if self.params["nb_gpus"] < 1:
            os.environ['CUDA_VISIBLE_DEVICES'] = ""
        if self.params["nb_gpus"] > 1:
            print("multiple gpus with TF not supported yet")
            return

        # if params["channels_first"]:
        #     keras.backend.set_image_data_format("channels_first")
        # else:
        #     keras.backend.set_image_data_format("channels_last")

        x_train, y_train = data

        y_train = to_categorical(y_train, num_classes=1000)


        if len(y_train.shape) > 1:
            cnt_classes = y_train.shape[1]
        else:
            cnt_classes = 1
        self.params["cnt_classes"] = cnt_classes
        model = self.net
        if self.params["mode"] != "training":
            raise NotADirectoryError("only training is implemented for TF")
        print("preheat")
        model.fit(x_train, y_train, batch_size=self.params["batch_size"], epochs=1)
        nb_epoch = 3
        print("train")
        start = timer()
        model.fit(x_train, y_train, batch_size=self.params["batch_size"], epochs=nb_epoch, verbose=1)
        end = timer()
        self.params["time"] = (end - start) / nb_epoch
        version_backend = tf.__version__
        # TODO: make this a nested dict
        # params["framework_full"] = "Keras-" + keras.__version__ + "/" + keras.backend.backend() + "_" + version_backend
        self.params["framework_full"] = "TensorFlow-" + version_backend
        return self.params


def run(params):
    backend_tf = DoTensorflow(params)
    return backend_tf.run()
