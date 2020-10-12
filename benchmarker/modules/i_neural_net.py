"""Module contains the interface for all deep learning modules"""
import argparse
import importlib
import os
import random

import numpy

from .power_mon import power_monitor_GPU, power_monitor_RAPL


class INeuralNet:
    """Interface for all deep learning modules"""

    def __init__(self, params, extra_args=None):
        self.params = params
        parser = argparse.ArgumentParser(description="Benchmark deep learning models")
        parser.add_argument("--mode", default="training")
        parser.add_argument("--nb_epoch", type=int, default=10)
        parser.add_argument("--random_seed", default=None)

        parsed_args, remaining_args = parser.parse_known_args(extra_args)

        params["mode"] = parsed_args.mode
        params["nb_epoch"] = parsed_args.nb_epoch
        assert params["mode"] in ["training", "inference"]
        params["path_out"] = os.path.join(params["path_out"], params["mode"])
        if "batch_size_per_device" not in params:
            self.params["batch_size_per_device"] = 32
        self.params["batch_size"] = self.params["batch_size_per_device"]
        if isinstance(params["problem"]["size"], int):
            params["problem"]["cnt_samples"] = params["problem"]["size"]
        else:
            params["problem"]["cnt_samples"] = params["problem"]["size"][0]
        assert params["problem"]["cnt_samples"] % params["batch_size"] == 0
        if self.params["nb_gpus"] > 0:
            self.params["batch_size"] = (
                self.params["batch_size_per_device"] * self.params["nb_gpus"]
            )
        if parsed_args.random_seed is not None:
            self.set_random_seed(int(parsed_args.random_seed))
        self.get_kernel(params, remaining_args)

    def get_kernel(self, params, remaining_args):
        """Default function to set `self.net`.  The derived do_* classes can
        override this function if there is some framework specific
        logic involved (e.g. GPU/TPU management etc).
        """
        path_params = f"benchmarker.modules.problems.{params['problem']['name']}.params"
        path_kernel = (
            f"benchmarker.modules.problems.{params['problem']['name']}."
            f"{params['framework']}"
        )
        # todo(vatai): combine tflite and tensorflow
        path_kernel = path_kernel.replace("tflite", "tensorflow")
        module_kernel = importlib.import_module(path_kernel)
        try:
            module_params = importlib.import_module(path_params)
            module_params.set_extra_params(params, remaining_args)
        except ImportError:
            assert remaining_args == []
        self.net = module_kernel.get_kernel(self.params)

    def load_data(self):
        params = self.params
        params["problem"]["cnt_batches_per_epoch"] = (
            params["problem"]["cnt_samples"] // self.params["batch_size"]
        )
        mod = importlib.import_module(
            "benchmarker.modules.problems." + params["problem"]["name"] + ".data"
        )
        get_data = getattr(mod, "get_data")
        data = get_data(params)
        params["problem"]["bytes_x_train"] = data[0].nbytes
        params["problem"]["size_sample"] = data[0][0].shape
        return data

    def set_random_seed(self, seed):
        """Default function to set random seeds which sets numpy and random
        modules seed.  This function should be overridden in the
        derived classes where this function should be called trough
        `super()` and also set the random seed of the framework.

        """
        # os.environ['PYTHONHASHSEED'] = '0' # this seems like too much
        numpy.random.seed(seed)
        random.seed(seed)

    def run(self):
        if self.params["nb_gpus"] > 0:
            power_monitor_gpu = power_monitor_GPU(self.params)
            power_monitor_gpu.start()
        power_monitor_cpu = power_monitor_RAPL(self.params)
        power_monitor_cpu.start()
        results = self.run_internal()
        if self.params["nb_gpus"] > 0:
            power_monitor_gpu.stop()
        power_monitor_cpu.stop()
        results["time_batch"] = (
            results["time_epoch"] / results["problem"]["cnt_batches_per_epoch"]
        )
        results["time_sample"] = results["time_batch"] / results["batch_size"]
        results["samples_per_second"] = (
            results["problem"]["cnt_samples"] / results["time_epoch"]
        )
        if results["power"]["joules_total"] > 0:
            results["samples_per_joule"] = (
                results["problem"]["cnt_samples"]
                * results["nb_epoch"]
                / self.params["power"]["joules_total"]
            )
        if "flop_estimated" in results["problem"]:
            results["flop_per_second_estimated"] = (
                results["problem"]["flop_estimated"] / results["time_total"]
            )
            results["gflop_per_second_estimated"] = results[
                "flop_per_second_estimated"
            ] / (1000 * 1000 * 1000)
        return results
