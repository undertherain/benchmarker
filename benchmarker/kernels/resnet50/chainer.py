import argparse
import chainercv

# Net = chainercv.links.ResNet50(pretrained_model="imagenet")


def get_kernel(params):
    return chainercv.links.ResNet50()
