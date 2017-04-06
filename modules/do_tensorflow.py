import os
from i_neural_net import INeuralNet


class DoTensorflow(INeuralNet):
    """docstring for ClassName"""
    def __init__(self, params):
        super().__init__()
        self.params = params
        
    def run(self, params, data):
        self.parameterize(params)
        os.environ["KERAS_BACKEND"] = "tensorflow"
        if params["nb_gpus"] < 1:
            os.environ['CUDA_VISIBLE_DEVICES'] = ""
        if params["nb_gpus"] > 1:
            print("multiple gpus with TF not supported yet")
            return
        from do_keras import run as run2
        params = run2(params, data)
        return params


def run(params, data):
    m = DoTensorflow(params)
    return m.run(params,data)
