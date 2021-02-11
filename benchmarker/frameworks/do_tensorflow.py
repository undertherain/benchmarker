# -*- coding: utf-8 -*-
"""TensorFlow support.
"""

import argparse
import os
from protonn.utils import describe_var
import numpy as np
from timeit import default_timer as timer

import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from .i_neural_net import INeuralNet


class Benchmark(INeuralNet):
    """docstring for ClassName"""

    def __init__(self, params, extra_args=None):
        parser = argparse.ArgumentParser(description="cf extra args")
        parser.add_argument("--precision", default="FP32")
        args, remaining_args = parser.parse_known_args(extra_args)
        params["problem"]["precision"] = args.precision
        assert params["problem"]["precision"] in ["FP32", "mixed"]
        super().__init__(params, remaining_args)
        self.params["channels_first"] = False
        os.environ["KERAS_BACKEND"] = "tensorflow"
        x_train, y_train = self.load_data()
        # Reshape from (nbatch, bs, ...) to (nbatch * bs, ...)
        # print(describe_var(x_train))
        self.x_train = np.vstack(x_train)
        self.y_train = np.vstack(y_train)
        # print(self.x_train.shape)
        if self.params["preheat"]:
            self.net.predict(self.x_train, self.params["batch_size"])

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
        if self.params["problem"]["precision"] == "mixed":
            policy = mixed_precision.Policy("mixed_float16")
            mixed_precision.set_policy(policy)
        with self.get_strategy().scope():
            super().get_kernel(module, remaining_args)

    def set_random_seed(self, seed):
        super().set_random_seed(seed)
        tf.random.set_seed(seed)

    def run(self):
        model = self.net
        nb_epoch = self.params["nb_epoch"]
        if self.params["mode"] == "training":
            start = timer()
            model.fit(
                self.x_train,
                self.y_train,
                batch_size=self.params["batch_size"],
                epochs=nb_epoch,
                verbose=1,
            )
        else:
            start = timer()
            for i in range(nb_epoch):
                model.predict(self.x_train, self.params["batch_size"], verbose=1)
        end = timer()
        self.params["time_total"] = end - start
        self.params["time_epoch"] = self.params["time_total"] / self.params["nb_epoch"]
        version_backend = tf.__version__
        # TODO: make this a nested dict
        # params["framework_full"] = "Keras-" + keras.__version__ + "/" + keras.backend.backend() + "_" + version_backend
        self.params["framework_full"] = "TensorFlow-" + version_backend
        return self.params
