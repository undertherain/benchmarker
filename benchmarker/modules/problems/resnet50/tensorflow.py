import keras
from keras.applications.resnet50 import ResNet50


def get_model(shape):
    model = ResNet50(weights=None)
    optimizer = keras.optimizers.Adam()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
    return model

Net = get_model