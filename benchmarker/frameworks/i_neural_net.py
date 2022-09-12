"""Module contains the interface for all deep learning modules"""
import argparse
# import os
import random

import numpy
from benchmarker.results import add_result_details

from .i_benchmark import IBenchmark


class INeuralNet(IBenchmark):
    """Interface for all deep learning modules"""

    def __init__(self, params, extra_args=None):
        self.params = params
        parser = argparse.ArgumentParser(description="Benchmark deep learning models")
        parser.add_argument("--mode", default="training")
        parser.add_argument("--nb_epoch", type=int, default=10)
        parser.add_argument("--random_seed", default=None)
        parser.add_argument("--batch_size", default=32)

        parsed_args, remaining_args = parser.parse_known_args(extra_args)
        params["batch_size"] = int(parsed_args.batch_size)
        params["batch_size_per_device"] = int(parsed_args.batch_size)
        params["mode"] = parsed_args.mode
        params["nb_epoch"] = parsed_args.nb_epoch
        assert params["mode"] in ["training", "inference"]
        params["path_ext"] = params["mode"]
        self.params["batch_size"] = self.params["batch_size_per_device"]
        problem_size = params["problem"]["size"]
        if isinstance(problem_size, int):
            cnt_samples = problem_size
        else:
            cnt_samples = problem_size[0]
        batch_size = params["batch_size"]
        assert (
            cnt_samples % batch_size == 0
        ), f"cnt_samples={cnt_samples} should be a multiple of batch_size={batch_size}"
        params["problem"]["cnt_samples"] = cnt_samples
        params["problem"]["cnt_batches_per_epoch"] = cnt_samples // batch_size

        if self.params["nb_gpus"] > 0:
            self.params["batch_size"] = (
                self.params["batch_size_per_device"] * self.params["nb_gpus"]
            )
        if parsed_args.random_seed is not None:
            self.set_random_seed(int(parsed_args.random_seed))
        self.get_kernel(params, remaining_args)

    def set_random_seed(self, seed):
        """Default function to set random seeds which sets numpy and random
        modules seed.  This function should be overridden in the
        derived classes where this function should be called trough
        `super()` and also set the random seed of the framework.

        """
        # os.environ['PYTHONHASHSEED'] = '0' # this seems like too much
        numpy.random.seed(seed)
        random.seed(seed)

    def post_process(self):
        results = self.params
        results["time_batch"] = (
            results["time_epoch"] / results["problem"]["cnt_batches_per_epoch"]
        )
        results["time_sample"] = results["time_batch"] / results["batch_size"]
        results["samples_per_second"] = (
            results["problem"]["cnt_samples"] / results["time_epoch"]
        )
        add_result_details(results)
