# Adapted from https://github.com/sparticlesteve/cosmoflow-benchmark/blob/master/models/cosmoflow_v1.py

"""Model specification for CosmoFlow This module contains the v1
implementation of the benchmark model.  It is deprecated now and being
replaced with the updated, more configurable architecture currently
defined in cosmoflow.py.

"""

import argparse
import ast

import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.optimizers import SGD


def scale_1p2(x):
    """Simple scaling function for Lambda layers.

    Just multiplies the input by 1.2. Useful for extending the coverage of a
    tanh activation for targets in the range [-1,1].
    """
    return x * 1.2


def proc_params(params, unparsed_args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_shape", default="128, 128, 128, 4")
    parser.add_argument("--target_size", default=4)
    parser.add_argument("--dropout", default=0)
    args, unparsed = parser.parse_known_args(unparsed_args)
    params["input_shape"] = ast.literal_eval(args.input_shape)
    params["target_size"] = args.target_size
    params["dropout"] = args.dropout
    assert unparsed == []


def build_model(input_shape, target_size, dropout=0):
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


def get_kernel(params, unparsed_args):
    """Construct the CosmoFlow 3D CNN model"""

    proc_params(params, unparsed_args)
    model = build_model(params["input_shape"], params["target_size"], params["dropout"])
    model.compile(optimizer=SGD())
    return model
