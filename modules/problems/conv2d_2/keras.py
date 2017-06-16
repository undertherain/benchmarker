import keras
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D


def get_model(params):
    model = keras.models.Sequential()
    if params["channels_first"]:
        data_format = "channels_first"
    else:
        data_format = "channels_last"

    model.add(Conv2D(filters=64, kernel_size=(2, 2),  padding='same', data_format=data_format, kernel_initializer="glorot_uniform", input_shape=params["shape_x_train"][1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=64, kernel_size=(2, 2),  padding='same', data_format=data_format, kernel_initializer="glorot_uniform"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(2, 2),  padding='same', data_format=data_format, kernel_initializer="glorot_uniform"))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=64, kernel_size=(2, 2),  padding='same', data_format=data_format, kernel_initializer="glorot_uniform"))
    model.add(Activation('relu'))
#    model.add(GlobalMaxPooling3D())
    model.add(Flatten())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    optimizer = keras.optimizers.Adam()
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=["accuracy"])
    return model
