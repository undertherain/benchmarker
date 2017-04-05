from do_keras import run as run2
import os


def run(params, data):
    if params["nb_gpus"] > 1:
        print("multiple gpus with Theano not supported yet")
        return
    if params["nb_gpus"] > 0:
        os.environ['THEANO_FLAGS'] = "device=cuda1"
    else:
        os.environ['THEANO_FLAGS'] = "device=cpu"
    os.environ["KERAS_BACKEND"] = "theano"
    params = run2(params, data)
    return params
