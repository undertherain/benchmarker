from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.optimizers import SGD


def get_kernel(params):
    model = ResNet50(weights=None)
    optimizer = SGD()
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"],
    )
    return model
