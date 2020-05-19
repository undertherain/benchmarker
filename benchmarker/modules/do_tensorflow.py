# -*- coding: utf-8 -*-
"""TensorFlow support.
"""

import os
from timeit import default_timer as timer

import tensorflow as tf

from .i_neural_net import INeuralNet


class Benchmark(INeuralNet):
    """docstring for ClassName"""

    def __init__(self, params, remaining_args=None):
        super().__init__(params, remaining_args)
        self.params["channels_first"] = False

    def get_tpu_addr(self):
        """Return grpc address if COLAB_TPU_ADDR is in the environment,
        otherwise None.

        Can be used to check if a TPU is present."""
        colab_tpu_addr = "COLAB_TPU_ADDR"
        if colab_tpu_addr not in os.environ:
            return None
        return "grpc://" + os.environ[colab_tpu_addr]

    def get_kernel(self, module, remaining_args):
        """
        Custom TF `get_kernel` method to handle TPU if
        available. https://www.tensorflow.org/guide/tpu
        """
        addr = self.get_tpu_addr()
        if addr:
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=addr)
            tf.config.experimental_connect_to_cluster(resolver)
            tf.tpu.experimental.initialize_tpu_system(resolver)
            strategy = tf.distribute.experimental.TPUStrategy(resolver)
            with strategy.scope():
                super().get_kernel(module, remaining_args)
        else:

            super().get_kernel(module, remaining_args)

    def run_internal(self):

        os.environ["KERAS_BACKEND"] = "tensorflow"
        if self.params["nb_gpus"] < 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        if self.params["nb_gpus"] > 1:
            print("multiple gpus with TF not supported yet")
            return

        # if params["channels_first"]:
        #     keras.backend.set_image_data_format("channels_first")
        # else:
        #     keras.backend.set_image_data_format("channels_last")

        # todo set image format
        data = self.load_data()
        x_train, y_train = data
        # Reshape from (nbatch, bs, ...) to (nbatch * bs, ...)
        x_train = x_train.reshape((-1,) + x_train.shape[-3:])
        y_train = y_train.reshape((-1,))

        y_train = tf.keras.utils.to_categorical(y_train, num_classes=1000)

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
        model.fit(
            x_train,
            y_train,
            batch_size=self.params["batch_size"],
            epochs=nb_epoch,
            verbose=1,
        )
        end = timer()
        self.params["time"] = (end - start) / nb_epoch
        version_backend = tf.__version__
        # TODO: make this a nested dict
        # params["framework_full"] = "Keras-" + keras.__version__ + "/" + keras.backend.backend() + "_" + version_backend
        self.params["framework_full"] = "TensorFlow-" + version_backend
        return self.params
