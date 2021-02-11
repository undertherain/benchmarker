import keras
from keras.layers import Input, Dense, Dropout, Activation, Flatten, ZeroPadding2D
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers import Convolution3D, MaxPooling3D, GlobalMaxPooling3D


def get_model(params):
    model = keras.models.Sequential()
    model.add(Convolution3D(32, 2, 2, 2, border_mode='same', init="glorot_uniform", input_shape=params["problem"]["shape_x_train"][1:]))
    model.add(Activation('relu'))
#    model.add(MaxPooling3D(pool_size=(2, 2 , 2)))
    model.add(Convolution3D(32, 2, 2, 2, border_mode='same', init="glorot_uniform"))
    model.add(Activation('relu'))
#    model.add(GlobalMaxPooling3D())
    model.add(Flatten())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    optimizer = keras.optimizers.Adam()
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=["accuracy"])
    return model
