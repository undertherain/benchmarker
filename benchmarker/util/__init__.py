import os


def get_script_dir():
    path = os.path.dirname(os.path.realpath(__file__))
    return path
