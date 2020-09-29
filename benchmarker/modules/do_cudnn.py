"""Module contains the interface for all deep learning modules"""
import importlib
from pathlib import Path

from benchmarker.util.abstractprocess import Process


class Benchmark:
    """Interface for all deep learning modules"""

    def __init__(self, params, remaining_args=None):
        path_params = f"benchmarker.modules.problems.{params['problem']['name']}.params"
        try:
            module_params = importlib.import_module(path_params)
            module_params.set_extra_params(params, remaining_args)
        except ImportError:
            assert remaining_args == []
        assert params["problem"]["name"] == "conv", (
            "only conv problem is defined for this framework, "
            f"not {params['problem']['name']}"
        )
        assert (
            "nb_gpus" in params and params["nb_gpus"] == 1
        ), "cuDNN requires exactly one GPU"
        self.params = params

    def run(self):
        bin_path = f"problems/{self.params['problem']['name']}/main"
        cmd_binary = Path(__file__).parent.joinpath(bin_path)
        if not cmd_binary.is_file():
            raise (RuntimeError(f"{cmd_binary} not found, run make manually"))

        problem = self.params["problem"]
        command = [cmd_binary]
        command.append(self.params["gpus"][0])
        command.append(problem["cudnn_conv_algo"])
        command.append(problem["nb_dims"])
        command.append(problem["cnt_samples"])
        command.append(problem["input_channels"])
        command.append(problem["cnt_filters"])
        command += problem["input_size"]
        command += problem["size_kernel"]
        command += problem["stride"]
        command += problem["dilation"]
        command += problem["padding"]
        command = list(map(str, command))

        process = Process(command=command)
        result = process.get_output()
        print(result)
        assert result["returncode"] == 0 and result["err"] == "", (
            "Error from binary!\n"
            f"retcode: {result['returncode']}\nstderr: {result['err']}"
        )
        elapsed_time = float(result["out"].strip())
        self.params["time"] = elapsed_time
        # self.params["GFLOP/sec"] = self.params["GFLOP"] / elapsed_time
