import argparse
from mxnet.gluon.model_zoo import vision


def get_kernel(params):
    return vision.get_model("resnet50_v1")
