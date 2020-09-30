from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.optimizers import SGD


def get_kernel(params):
    model = VGG19(weights=None)
    optimizer = SGD()
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"],
    )
    return model
