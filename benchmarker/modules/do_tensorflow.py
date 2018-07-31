# -*- coding: utf-8 -*-
"""TensorFlow support.
"""

import os
from .i_neural_net import INeuralNet


class DoTensorflow(INeuralNet):
    """docstring for ClassName"""

    def __init__(self, params):
        super().__init__(params)
        self.params["channels_first"] = False

    def run(self):
        # todo set image format
        data = self.load_data()

        os.environ["KERAS_BACKEND"] = "tensorflow"
        if self.params["nb_gpus"] < 1:
            os.environ['CUDA_VISIBLE_DEVICES'] = ""
        if self.params["nb_gpus"] > 1:
            print("multiple gpus with TF not supported yet")
            return
        from ._keras import run as run2
        params = run2(self.params, data)
        return params


def run(params):
    backend_tf = DoTensorflow(params)
    return backend_tf.run()
