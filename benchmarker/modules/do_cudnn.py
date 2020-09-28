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

        batch_size, in_chan, out_chan = 1, 3, 3
        in_dims = [578, 549]
        ker_dims = [3, 3]
        ker_pads = [1, 1]
        ker_stride = [1, 1]
        ker_dilation = [1, 1]

        problem = self.params["problem"]
        command = [cmd_binary]
        command.append(self.params["gpus"][0])
        command.append(problem["cudnn_conv_algo"])
        command.append(problem["nb_dims"])
        command += [batch_size, in_chan, out_chan]
        command += in_dims
        command += ker_dims
        command += ker_pads
        command += ker_stride
        command += ker_dilation
        command = list(map(str, command))

        print(self.params)
        process = Process(command=command)
        result = process.get_output()
        std_out = result["out"]
        print(std_out.strip())
        elapsed_time = float(std_out.strip())
        self.params["time"] = elapsed_time
        # self.params["GFLOP/sec"] = self.params["GFLOP"] / elapsed_time
