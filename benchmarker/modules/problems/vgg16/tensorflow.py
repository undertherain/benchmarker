from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import SGD


def get_kernel(parsm, unparsed_args):
    model = VGG16(weights=None)
    optimizer = SGD()
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=["accuracy"])
    return model
