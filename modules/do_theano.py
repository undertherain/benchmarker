import os
from i_neural_net import INeuralNet


class DoTheano(INeuralNet):

    def __init__(self, params):
        super().__init__()
        self.params = params

    def run(self, params, data):
        self.parameterize(params)
        if params["nb_gpus"] > 1:
            print("multiple gpus with Theano not supported yet")
            return
        if params["nb_gpus"] > 0:
            os.environ['THEANO_FLAGS'] = "device=cuda1"
        else:
            os.environ['THEANO_FLAGS'] = "device=cpu"
        os.environ["KERAS_BACKEND"] = "theano"
        from do_keras import run as run2
        params = run2(params, data)
        return params


def run(params, data):
    m = DoTheano(params)
    return m.run(params, data)
