import os
from do_keras import run as run2


def run(params, data):
    os.environ["KERAS_BACKEND"] = "tensorflow"
    if params["nb_gpus"] < 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ""
    if params["nb_gpus"] > 1:
        print("multiple gpus with TF not supported yet")
        return
    params = run2(params, data)
    params
