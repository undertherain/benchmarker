"""Module contains the interface for all deep learning modules"""
import argparse
import os

from benchmarker.util.abstractprocess import Process


class Benchmark:
    """Interface for all deep learning modules"""

    def __init__(self, params, remaining_args=None):
        self.params = params
        # parser = argparse.ArgumentParser(description="gemm extra args")
        # parser.add_argument("--precision", default="FP32")
        # args, remaining_args = parser.parse_known_args(remaining_args)
        # args = parser.parse_args(remaining_args)
        # params["problem"]["precision"] = args.precision
        # params["path_out"] = os.path.join(
        #     params["path_out"], params["problem"]["precision"]
        # )

        params["problem"]["size"] = 42  # self.matrix_size

        if params["problem"]["name"] != "conv":
            raise Exception(
                "only conv problem is defined for this framework, "
                f"not {params['problem']['name']}"
            )

    def run(self):
        if "nb_gpus" in self.params:
            if self.params["nb_gpus"] == 0:
                raise Exception("cudnn requires at least one GPU")
        # size = " ".join(map(str, self.params["problem"]["size"]))
        path_binary = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "problems/cudnn/main"
        )
        if not os.path.isfile(path_binary):
            raise (RuntimeError(f"{path_binary} not found, run make manually"))
        command = [path_binary, "tf.png", "1"]
        process = Process(command=command)
        result = process.get_output()
        std_out = result["out"]
        print(std_out.strip())
        elapsed_time = float(std_out.strip())
        self.params["time"] = elapsed_time
        # self.params["GFLOP/sec"] = self.params["GFLOP"] / elapsed_time
