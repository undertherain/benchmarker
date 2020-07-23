import numpy as np
import torch.nn as nn

from ..helpers_torch import Regression
from .params import proc_params


class scale_1p2(nn.Module):
    def forward(self, input):
        return 1.2 * input


def build_model(input_shape, target_size, dropout=0):
    shape = np.array(input_shape[:-1], dtype=np.int)
    conv_args = {
        "in_channels": input_shape[-1],
        "out_channels": 16,
        "kernel_size": 2,
    }
    maxpool_args = dict(kernel_size=2)

    layers = [
        nn.Conv3d(**conv_args),
        nn.LeakyReLU(),
        nn.MaxPool3d(**maxpool_args),
    ]
    shape = (shape - 1) // 2

    conv_args["in_channels"] = 16
    for _ in range(4):
        layers += [
            nn.Conv3d(**conv_args),
            nn.LeakyReLU(),
            nn.MaxPool3d(**maxpool_args),
        ]
        shape = (shape - 1) // 2

    flat_shape = np.prod(shape) * 16
    layers += [
        nn.Flatten(),
        nn.Dropout(dropout),
        #
        nn.Linear(flat_shape, 128),
        nn.LeakyReLU(),
        nn.Dropout(),
        #
        nn.Linear(128, 64),
        nn.LeakyReLU(),
        nn.Dropout(),
        #
        nn.Linear(64, target_size),
        nn.Tanh(),
        scale_1p2(),
    ]

    return nn.Sequential(*layers)


def get_kernel(params, unparsed_args):
    """Construct the CosmoFlow 3D CNN model"""

    proc_params(params, unparsed_args)
    net = build_model(params["input_shape"], params["target_size"], params["dropout"])
    return Regression(params, net)
