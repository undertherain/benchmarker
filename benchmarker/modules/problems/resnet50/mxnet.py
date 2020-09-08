import argparse
from mxnet.gluon.model_zoo import vision


def get_kernel(params):
    parser = argparse.ArgumentParser(description='Benchmark resnet50')
    parser.parse_args(unparsed_args)
    return vision.get_model("resnet50_v1")
