from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.optimizers import SGD


def get_kernel(params, unparsed_args):
    model = Xception(weights=None)
    optimizer = SGD()
    model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )
    return model
