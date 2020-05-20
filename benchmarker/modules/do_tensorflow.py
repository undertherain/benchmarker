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

    def get_strategy(self):
        try:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
        except ValueError:
            tpu = None
            gpus = tf.config.experimental.list_logical_devices("GPU")
        if tpu:
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            strategy = tf.distribute.experimental.TPUStrategy(tpu)
            worker_str = tpu.cluster_spec().as_dict()["worker"]
            rep = strategy.num_replicas_in_sync
            self.params["device"] = "COLAB TPU"
            self.params["tpus"] = {
                "worker_srt": worker_str,
                "num_replicas_in_sync": rep,
            }
        elif len(gpus) > 1:  # multiple GPUs in one VM
            strategy = tf.distribute.MirroredStrategy(gpus)
        else:  # default strategy that works on CPU and single GPU
            strategy = tf.distribute.get_strategy()
        return strategy

    def get_kernel(self, module, remaining_args):
        """
        Custom TF `get_kernel` method to handle TPU if
        available. https://www.tensorflow.org/guide/tpu
        """
        with self.get_strategy().scope():
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
