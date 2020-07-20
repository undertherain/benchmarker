# Adapted from https://github.com/sparticlesteve/cosmoflow-benchmark/blob/master/models/cosmoflow_v1.py

"""Model specification for CosmoFlow This module contains the v1
implementation of the benchmark model.  It is deprecated now and being
replaced with the updated, more configurable architecture currently
defined in cosmoflow.py.

"""

import tensorflow as tf
import tensorflow.keras.layers as layers


def scale_1p2(x):
    """Simple scaling function for Lambda layers.

    Just multiplies the input by 1.2. Useful for extending the coverage of a
    tanh activation for targets in the range [-1,1].
    """
    return x * 1.2


def get_kernel(params, unparsed_args):
    """Construct the CosmoFlow 3D CNN model"""

    ### !!! STUB INPUTS/VARIABLES!!! ###
    input_shape = 0  # default: [128, 128, 128, 4]
    ### !!! STUB INPUTS/VARIABLES!!! ###
    target_size = 0  # default: 4
    ### !!! STUB INPUTS/VARIABLES!!! ###
    dropout = 0  # default 0

    conv_args = dict(kernel_size=2, padding="valid")

    model = tf.keras.models.Sequential(
        [
            layers.Conv3D(16, input_shape=input_shape, **conv_args),
            layers.LeakyReLU(),
            layers.MaxPool3D(pool_size=2),
            #
            layers.Conv3D(16, **conv_args),
            layers.LeakyReLU(),
            layers.MaxPool3D(pool_size=2),
            #
            layers.Conv3D(16, **conv_args),
            layers.LeakyReLU(),
            layers.MaxPool3D(pool_size=2),
            #
            layers.Conv3D(16, **conv_args),
            layers.LeakyReLU(),
            layers.MaxPool3D(pool_size=2),
            #
            layers.Conv3D(16, **conv_args),
            layers.LeakyReLU(),
            layers.MaxPool3D(pool_size=2),
            #
            layers.Flatten(),
            layers.Dropout(dropout),
            #
            layers.Dense(128),
            layers.LeakyReLU(),
            layers.Dropout(dropout),
            #
            layers.Dense(64),
            layers.LeakyReLU(),
            layers.Dropout(dropout),
            #
            layers.Dense(target_size, activation="tanh"),
            layers.Lambda(scale_1p2),
        ]
    )

    return model
