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
        if isinstance(params["problem"]["size"], int):
            M = N = K = params["problem"]["size"]
        else:
            M, N, K = params["problem"]["size"]
        self.matrix_size = M, N, K
        params["problem"]["size"] = self.matrix_size
        gflop = (2.0 * M * N * K) / (1024 ** 3)
        params["GFLOP"] = gflop
        if params["problem"]["name"] != "gemm":
            raise Exception("only gemm problem is defined for this framework")
