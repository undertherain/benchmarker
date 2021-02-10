from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.optimizers import SGD


def get_kernel(params):
    model = Xception(weights=None)
    optimizer = SGD()
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"],
    )
    return model
