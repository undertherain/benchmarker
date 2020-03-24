import argparse
import chainercv

# Net = chainercv.links.ResNet50(pretrained_model="imagenet")


def get_kernel(params, unparsed_args):
    parser = argparse.ArgumentParser(description='Benchmark resnet50')
    parser.parse_args(unparsed_args)
    return chainercv.links.ResNet50()
