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
        gpus = params["gpus"]
        super().__init__(params, remaining_args)
        self.params["channels_first"] = False
        os.environ["KERAS_BACKEND"] = "tensorflow"

    def get_strategy(self):
        gpu_count_same = self.params["nb_gpus"] == len(
            tf.config.list_physical_devices("GPU")
        )
        assert gpu_count_same, "Tensorflow not compiled with GPU support"
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
        elif len(gpus) > 1:
            strategy = tf.distribute.MirroredStrategy(gpus)
        elif self.params["nb_gpus"] == 1:
            strategy = tf.distribute.get_strategy()
        else:  # Make sure we run on CPU
            strategy = tf.distribute.get_strategy()

        return strategy

    def get_kernel(self, module, remaining_args):
        """
        Custom TF `get_kernel` method to handle TPU if
        available. https://www.tensorflow.org/guide/tpu
        """
        with self.get_strategy().scope():
            super().get_kernel(module, remaining_args)

    def set_random_seed(self, seed):
        super().set_random_seed(seed)
        tf.random.set_seed(seed)

    def run_internal(self):

        # if params["channels_first"]:
        #     keras.backend.set_image_data_format("channels_first")
        # else:
        #     keras.backend.set_image_data_format("channels_last")

        # todo set image format
        x_train, y_train = self.load_data()
        # Reshape from (nbatch, bs, ...) to (nbatch * bs, ...)
        x_train = x_train.reshape((-1,) + x_train.shape[2:])
        y_train = y_train.reshape((-1,) + y_train.shape[2:])

        y_train = tf.keras.utils.to_categorical(y_train, num_classes=1000)

        if len(y_train.shape) > 1:
            cnt_classes = y_train.shape[1]
        else:
            cnt_classes = 1
        self.params["cnt_classes"] = cnt_classes
        model = self.net
        nb_epoch = self.params["nb_epoch"]
        bs = self.params["batch_size"]
        if self.params["mode"] == "training":
            print("preheat")
            model.fit(x_train, y_train, batch_size=bs, epochs=1)
            print("train")
            start = timer()
            model.fit(x_train, y_train, batch_size=bs, epochs=nb_epoch, verbose=1)
        else:
            # preheat
            model.predict(x_train, bs)
            start = timer()
            model.predict(x_train, bs, verbose=1)
        end = timer()
        self.params["time"] = (end - start) / nb_epoch
        version_backend = tf.__version__
        # TODO: make this a nested dict
        # params["framework_full"] = "Keras-" + keras.__version__ + "/" + keras.backend.backend() + "_" + version_backend
        self.params["framework_full"] = "TensorFlow-" + version_backend
        return self.params
