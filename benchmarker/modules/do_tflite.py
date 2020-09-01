# -*- coding: utf-8 -*-
"""'Framework' for edge-tpu devices.

Started from 

  https://coral.ai/docs/accelerator/get-started/, 

which uses code from

  https://github.com/google-coral/tflite/python/examples/classification

Installed tflite_runtime following these instructions

  https://www.tensorflow.org/lite/guide/python

"""

import os
import platform
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

    def get_kernel(self, module, remaining_args):
        """
        Custom TF `get_kernel` method to handle TPU if
        available. https://www.tensorflow.org/guide/tpu
        """
        super().get_kernel(module, remaining_args)
        # todo(vatai): figure a nicer way to get input shape
        x_train, _ = self.load_data()
        x_train = x_train.reshape((-1,) + x_train.shape[2:])
        self.net.build(x_train.shape)
        converter = tf.lite.TFLiteConverter.from_keras_model(self.net)
        self.net = converter.convert()

    def set_random_seed(self, seed):
        super().set_random_seed(seed)
        tf.random.set_seed(seed)

    def _make_interpreter(self):
        shared_lib = {
            "Linux": "libedgetpu.so.1",
            "Darwin": "libedgetpu.1.dylib",
            "Windows": "edgetpu.dll",
        }[platform.system()]
        delegate = tf.lite.experimental.load_delegate(shared_lib, {})
        return tf.lite.Interpreter(
            model_content=self.net,
            experimental_delegates=[delegate],
        )

    def run_internal(self):
        assert self.params["mode"] == "inference", "Only inference supported ATM"

        x_train, y_train = self.load_data()
        x_train = x_train.reshape((-1,) + x_train.shape[2:])
        y_train = y_train.reshape((-1,) + y_train.shape[2:])

        model = self.net
        bs = self.params["batch_size"]
        # preheat

        interpreter = self._make_interpreter()
        interpreter.allocate_tensors()
        # set input
        tensor_index = interpreter.get_input_details()[0]["index"]
        interpreter.tensor(tensor_index)()[:] = x_train
        start = timer()
        interpreter.invoke()
        end = timer()
        # optionally get output?
        self.params["time_total"] = end - start
        self.params["time_epoch"] = self.params["time_total"] / self.params["nb_epoch"]
        version_backend = tf.__version__

        self.params["framework_full"] = "TFlite-" + version_backend
        return self.params
