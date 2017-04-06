import keras
from keras.layers import Input, Dense, Dropout, Activation, Flatten, ZeroPadding2D
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D

def get_model(shape):
    model=keras.models.Sequential()
    model.add(Conv2D(filters = 64, kernel_size=(2, 2),  padding='same', data_format="channels_first",  kernel_initializer="glorot_uniform", input_shape=shape ))
    model.add(Activation('relu'))
    model.add(Conv2D(filters = 64, kernel_size=(2, 2),  padding='same', data_format="channels_first",  kernel_initializer="glorot_uniform"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2 , 2)))
    model.add(Conv2D(filters = 64, kernel_size=(2, 2),  padding='same', data_format="channels_first",  kernel_initializer="glorot_uniform"))
    model.add(Activation('relu'))
    model.add(Conv2D(filters = 64, kernel_size=(2, 2),  padding='same', data_format="channels_first",  kernel_initializer="glorot_uniform"))
    model.add(Activation('relu'))
#    model.add(GlobalMaxPooling3D())
    model.add(Flatten())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model
