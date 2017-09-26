import keras
from keras.applications.vgg16 import VGG16


def get_model(shape):
    model = VGG16(weights=None)
    optimizer = keras.optimizers.Adam()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
    return model
