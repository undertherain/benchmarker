from keras.applications.vgg16 import VGG16


def get_model(shape):
    model = VGG16(weights=None)
    return model
