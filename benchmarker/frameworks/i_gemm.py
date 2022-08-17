"""Module contains the interface for all gemm learning modules"""
import argparse

from benchmarker.results import add_result_details

from .i_binary import IBinary


class IGEMM(IBinary):
    """Interface for all gemm learning modules"""

    def __init__(self, params, remaining_args=None):
        super().__init__(params, remaining_args)

        prec_list = self.expected_precisions()
        precision = self.params["problem"]["precision"]
        assert precision in prec_list, f"Expected precisions: {prec_list}"

        cnt_gpus_requested = self.params["nb_gpus"] if "nb_gpus" in self.params else 0
        need_gpus = self.need_gpus()
        msg = (
            f"{self.params['framework']} requires GPUs"
            if 'framework' in self.params
            else "GPUs required"
        )

        assert not need_gpus or cnt_gpus_requested > 0, msg

        self.data = self.load_data()
        if params["problem"]["name"] not in ["gemm", "batchmatmul"]:
            raise Exception(
                f"only gemm problem is defined for this framework, "
                f"{params['problem']['name']} is not!"
            )

    def expected_precisions(self):
        return ["FP32", "mixed"]

    def need_gpus(self):
        return False

    def process_params(self, remaining_args):
        remaining_args = super().process_params(remaining_args)
        parser = argparse.ArgumentParser(description="gemm extra args")
        parser.add_argument("--precision", default="FP32")
        parser.add_argument("--nb_epoch", type=int, default=1)
        # args, remaining_args = parser.parse_known_args(remaining_args)
        # parser = argparse.ArgumentParser(description='Benchmark GEMM operations')
        args, remaining_args = parser.parse_known_args(remaining_args)
        self.params["problem"]["precision"] = args.precision
        self.params["nb_epoch"] = args.nb_epoch
        self.params["path_ext"] = self.params["problem"]["precision"]
        return remaining_args

    def post_process(self):
        gflop = self.params["problem"]["flop_estimated"] / (1000 ** 3)
        self.params["problem"]["gflop_estimated"] = gflop
        add_result_details(self.params)
        self.params["time_epoch"] = self.params["time_total"] / self.params["nb_epoch"]
