from timeit import default_timer as timer
import keras
import keras.models
import importlib


def run(params, data):
    X_train, Y_train = data
    mod = importlib.import_module("problems." + params["problem"]+".keras")
    get_model = getattr(mod, 'get_model')

    if len(Y_train.shape) > 1:
        cnt_classes = Y_train.shape[1]
    else:
        cnt_classes = 1
    params["cnt_classes"] = cnt_classes
    model = get_model(params)
    print("preheat")
    model.fit(X_train, Y_train, batch_size=params["batch_size"], epochs=1)
    nb_epoch = 3
    print("train")
    start = timer()
    model.fit(X_train, Y_train, batch_size=params["batch_size"], epochs=nb_epoch, verbose=1)
    end = timer()
    params["time"] = (end-start)/nb_epoch
    if params["framework"] == "theano":
        import theano
        version_backend = theano.__version__
    else:
        import tensorflow as tf
        version_backend = tf.__version__
    params["framework_full"] = "Keras-" + keras.__version__ + "/" + keras.backend.backend() + "_" + version_backend
    return params
