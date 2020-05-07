import argparse
import torchvision.models as models


def get_kernel(params, unparsed_args):
    parser = argparse.ArgumentParser(description="Benchmark resnet50")
    parser.parse_args(unparsed_args)
    return models.vgg16()
