"""Module contains the interface for all gemm learning modules"""
import argparse
import os
from .ops import detalize_ops_results
from .i_benchmark import IBenchmark


class IGEMM(IBenchmark):
    """Interface for all gemm learning modules"""

    def __init__(self, params, extra_args=None):
        print("WE ARE HERE")
        self.params = params
        parser = argparse.ArgumentParser(description="gemm extra args")
        parser.add_argument("--precision", default="FP32")
        parser.add_argument("--nb_epoch", type=int, default=1)
        # args, remaining_args = parser.parse_known_args(remaining_args)
        # parser = argparse.ArgumentParser(description='Benchmark GEMM operations')
        args, remaining_args = parser.parse_known_args(extra_args)
        params["problem"]["precision"] = args.precision
        params["nb_epoch"] = args.nb_epoch
        params["path_out"] = os.path.join(params["path_out"],
                                          params["problem"]["precision"])
        super().__init__(params, remaining_args)
        # TODO: this should also go up
        self.data = self.load_data()
        if params["problem"]["name"] not in ["gemm", "batchmatmul"]:
            raise Exception(f"only gemm problem is defined for this framework, not {params['problem']['name']}")

    def post_process(self):
        self.params["problem"]["gflop_estimated"] = self.params["problem"]["flop_estimated"] / (1000 ** 3)
        detalize_ops_results(self.params)
        self.params["time_epoch"] = self.params["time_total"] / self.params["nb_epoch"]
