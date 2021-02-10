import os
from .i_neural_net import INeuralNet


class DoTheano(INeuralNet):

    def __init__(self, params):
        super().__init__(params)
        self.params["channels_first"] = True

    def run(self):
        data = self.load_data()
        if self.params["nb_gpus"] > 1:
            print("multiple gpus with Theano not supported yet")
            return
        if self.params["nb_gpus"] > 0:
            os.environ['THEANO_FLAGS'] = "device=cuda0"
        else:
            os.environ['THEANO_FLAGS'] = "device=cpu"
        os.environ["KERAS_BACKEND"] = "theano"
        from ._keras import run as run2
        params = run2(self.params, data)
        return params


def run(params):
    backend_theano = DoTheano(params)
    return backend_theano.run()
