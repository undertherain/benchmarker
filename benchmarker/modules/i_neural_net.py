"""Module contains the interface for all deep learning modules"""
import argparse
import importlib
import os


class INeuralNet:
    """Interface for all deep learning modules"""

    def __init__(self, params, extra_args=None):
        parser = argparse.ArgumentParser(description="Benchmark deep learning models")
        parser.add_argument("--mode", default="training")
        parsed_args, remaining_args = parser.parse_known_args(extra_args)
        params["mode"] = parsed_args.mode
        params["path_out"] = os.path.join(params["path_out"], params["mode"])
        self.params = params
        if "batch_size_per_device" in params:
            self.params["batch_size_per_device"] = params["batch_size_per_device"]
        else:
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
        self.params["channels_first"] = True
        if params["mode"] is None:
            params["mode"] = "training"
        assert params["mode"] in ["training", "inference"]
        path = f"benchmarker.modules.problems.{params['problem']['name']}.{params['framework']}"
        kernel_module = importlib.import_module(path)
        self.get_kernel(kernel_module, remaining_args)

    def get_kernel(self, module, remaining_args):
        """Default function to set `self.net`.  The derived do_* classes can
        override this function if there is some framework specific
        logic involved (e.g. GPU/TPU management etc).

        :param module: module which implements the framework specific
        `get_kernel` method.
        :param remaining_args: (Sub)problem dependent arguments.

        """
        self.net = module.get_kernel(self.params, remaining_args)

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

    def run(self):
        results = self.run_internal()
        results["time_epoch"] = results["time"]
        results["time_batch"] = (
            results["time_epoch"] / results["problem"]["cnt_batches_per_epoch"]
        )
        results["time_sample"] = results["time_batch"] / results["batch_size"]
        results["samples_per_second"] = (
            results["problem"]["cnt_samples"] / results["time_epoch"]
        )
        return results
