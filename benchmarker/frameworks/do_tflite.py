# -*- coding: utf-8 -*-
"""'Framework' for edge-tpu devices.

TL;DR: install the `libedgetpu-max` library and add yourself to the
`plugdev`. Something like

  sudo apt-get install libedgetpu1-max

  sudo usermod -aG plugdev $(whoami)

Don't forget to login/logout and plug-out/plug-in the egde-tpu device
(or restart the computer).

Started from example in

  https://coral.ai/docs/accelerator/get-started/

which uses code from

  https://github.com/google-coral/tflite/python/examples/classification

If your install `tflite_runtime` following these instructions

  https://www.tensorflow.org/lite/guide/python

you can test the `/python/examples/classification` examples if it works.

"""

import os
import platform
from timeit import default_timer as timer

import tensorflow as tf

from .i_neural_net import INeuralNet


class Benchmark(INeuralNet):
    """docstring for ClassName"""

    def __init__(self, params, remaining_args=None):
        params["channels_first"] = False
        super().__init__(params, remaining_args)
        # todo(vatai): set channels should be here but it is moved to
        # `get_kernel()`
        #
        # self.params["channels_first"] = False
        os.environ["KERAS_BACKEND"] = "tensorflow"

    def get_kernel(self, module, remaining_args):
        """
        Custom TF `get_kernel` method to handle TPU if
        available. https://www.tensorflow.org/guide/tpu
        """
        super().get_kernel(module, remaining_args)
        # todo(vatai): figure a nicer way to get input shape
        self.x_train, _ = self.load_data()
        shape = self.x_train[0].shape
        self.net.build(shape)
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
        try:
            delegate = tf.lite.experimental.load_delegate(shared_lib, {})
        except ValueError as err:
            print(f"ValueError: {err}\nTPU probably not plugged in.")
            exit(1)
        self.params["device"] = "Edge TPU"
        return tf.lite.Interpreter(
            model_content=self.net,
            experimental_delegates=[delegate],
        )

    def run(self):
        assert self.params["mode"] == "inference", "Only inference supported ATM"

        # preheat
        interpreter = self._make_interpreter()
        interpreter.allocate_tensors()
        # set input
        tensor_index = interpreter.get_input_details()[0]["index"]

        start = timer()
        for batch in range(self.params["problem"]["cnt_batches_per_epoch"]):
            interpreter.set_tensor(tensor_index, self.x_train[batch])
            interpreter.invoke()
        end = timer()

        # optionally get output?
        self.params["time_total"] = end - start
        self.params["time_epoch"] = self.params["time_total"] / self.params["nb_epoch"]
        version_backend = tf.__version__

        self.params["framework_full"] = "TFlite-" + version_backend
        return self.params
