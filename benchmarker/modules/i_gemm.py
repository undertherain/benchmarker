"""Module contains the interface for all deep learning modules"""
# import argparse
import os


class IGEMM():
    """Interface for all deep learning modules"""

    def __init__(self, params, remaining_args=None):
        self.params = params
        # parser = argparse.ArgumentParser(description='Benchmark GEMM operations')
        # parser.add_argument('--mode', default=None)
        # args = parser.parse_args(remaining_args)
        # TODO: read size from args
        # TODO: add float type as arg
        params["problem"]["precision"] = "FP32"
        params["path_out"] = os.path.join(params["path_out"],
                                          params["problem"]["precision"])
