"""Module contains the interface for all deep learning modules"""
import argparse
import os
from .ops import detalize_ops_results
from .i_benchmark import IBenchmark


class IGEMM(IBenchmark):
    """Interface for all gemm learning modules"""

    def __init__(self, params, remaining_args=None):
        self.params = params
        parser = argparse.ArgumentParser(description="gemm extra args")
        parser.add_argument("--precision", default="FP32")
        parser.add_argument("--nb_epoch", type=int, default=1)
        # args, remaining_args = parser.parse_known_args(remaining_args)
        # parser = argparse.ArgumentParser(description='Benchmark GEMM operations')
        args = parser.parse_args(remaining_args)
        params["problem"]["precision"] = args.precision
        params["nb_epoch"] = args.nb_epoch
        params["path_out"] = os.path.join(params["path_out"],
                                          params["problem"]["precision"])
        if isinstance(params["problem"]["size"], int):
            params["problem"]["size"] = [params["problem"]["size"]] * 3
        M, N, K = params["problem"]["size"]
        self.matrix_size = M, N, K
        params["problem"]["size"] = self.matrix_size
        flop = (2.0 * M * N * K)
        params["problem"]["flop_estimated"] = flop * self.params["nb_epoch"]
        params["problem"]["gflop_estimated"] = params["problem"]["flop_estimated"] / (1000 ** 3)
        if params["problem"]["name"] != "gemm":
            raise Exception(f"only gemm problem is defined for this framework, not {params['problem']['name']}")

    def post_process(self):
        detalize_ops_results(self.params)
        self.params["time_epoch"] = self.params["time_total"] / self.params["nb_epoch"]
