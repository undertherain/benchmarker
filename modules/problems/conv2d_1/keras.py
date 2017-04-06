import keras
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D


def get_model(shape):
    model=keras.models.Sequential()
    model.add(Conv2D(32, (2, 2),  padding='same', kernel_initializer="glorot_uniform", input_shape=shape ))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (2, 2),  padding='same', kernel_initializer="glorot_uniform"))
    model.add(Activation('relu'))
#    model.add(GlobalMaxPooling3D())
    model.add(Flatten())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model
