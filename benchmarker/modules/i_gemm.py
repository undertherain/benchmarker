"""Module contains the interface for all deep learning modules"""
import argparse
import os


class IGEMM():
    """Interface for all deep learning modules"""

    def __init__(self, params, remaining_args=None):
        self.params = params
        parser = argparse.ArgumentParser(description="gemm extra args")
        parser.add_argument("--precision", default="FP32")
        # args, remaining_args = parser.parse_known_args(remaining_args)
        # parser = argparse.ArgumentParser(description='Benchmark GEMM operations')
        args = parser.parse_args(remaining_args)
        params["problem"]["precision"] = args.precision
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
            raise Exception(f"only gemm problem is defined for this framework, not {params['problem']['name']}")
