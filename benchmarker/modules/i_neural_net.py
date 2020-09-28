"""Module contains the interface for all deep learning modules"""
import argparse
import importlib
import os
import random

import numpy
import threading
from time import sleep
import numpy as np
import pyRAPL


class INeuralNet:
    """Interface for all deep learning modules"""

    def __init__(self, params, extra_args=None):
        self.params = params
        parser = argparse.ArgumentParser(description="Benchmark deep learning models")
        parser.add_argument("--mode", default="training")
        parser.add_argument("--nb_epoch", type=int, default=10)
        parser.add_argument("--power_sampling_ms", type=int, default=100)
        parser.add_argument("--random_seed", default=None)

        parsed_args, remaining_args = parser.parse_known_args(extra_args)

        params["mode"] = parsed_args.mode
        params["power"] = {}
        params["power"]["sampling_ms"] = parsed_args.power_sampling_ms
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
        # self.params["channels_first"] = True
        if parsed_args.random_seed is not None:
            self.set_random_seed(int(parsed_args.random_seed))
        self.get_kernel(params, remaining_args)
        self.keep_monitor = True
        try:
            pyRAPL.setup()
            self.rapl_enabled = True
        except:
            self.rapl_enabled = False

    def get_kernel(self, params, remaining_args):
        """Default function to set `self.net`.  The derived do_* classes can
        override this function if there is some framework specific
        logic involved (e.g. GPU/TPU management etc).
        """
        path_params = f"benchmarker.modules.problems.{params['problem']['name']}.params"
        path_kernel = (f"benchmarker.modules.problems.{params['problem']['name']}."
                       f"{params['framework']}")
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
        self.params["power"]["joules_total"] = 0
        thread_monitor = threading.Thread(target=self.monitor, args=())
        thread_monitor.start()                                  # S
        if self.rapl_enabled:
            meter_rapl = pyRAPL.Measurement('bar')
            meter_rapl.begin()
        results = self.run_internal()
        self.keep_monitor = False
        if self.rapl_enabled:
            meter_rapl.end()
            self.params["power"]["joules_CPU"] = sum(meter_rapl.result.pkg) / 1000000.0
            self.params["power"]["joules_RAM"] = sum(meter_rapl.result.dram) / 1000000.0
        thread_monitor.join()
        if self.rapl_enabled:
            self.params["power"]["joules_total"] += self.params["power"]["joules_CPU"]
            self.params["power"]["joules_total"] += self.params["power"]["joules_RAM"]
        results["time_batch"] = (
            results["time_epoch"] / results["problem"]["cnt_batches_per_epoch"]
        )
        results["time_sample"] = results["time_batch"] / results["batch_size"]
        results["samples_per_second"] = (
            results["problem"]["cnt_samples"] / results["time_epoch"]
        )
        results["samples_per_joule_GPU"] = results["problem"]["cnt_samples"] * results["nb_epoch"] / self.params["power"]["joules_GPU"]
        if results["power"]["joules_total"] > 0:
            results["samples_per_joule"] = results["problem"]["cnt_samples"] * results["nb_epoch"] / self.params["power"]["joules_total"]
        if "flop_estimated" in results["problem"]:
            results["flop_per_second_estimated"] = results["problem"]['flop_estimated'] / results["time_total"]
            results["gflop_per_second_estimated"] = results["flop_per_second_estimated"] / (1000 * 1000 * 1000)
        return results

    def monitor(self):
        lst_power_gpu = []
        # TODO: move this to init
        # TODO: query multiple GPUs
        # TODO: don't do this if GPU is not used
        from py3nvml.py3nvml import nvmlInit, nvmlShutdown, nvmlDeviceGetCount
        from py3nvml.py3nvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetPowerUsage
        nvmlInit()
        cnt_gpu = nvmlDeviceGetCount()
        print(f"##CNT GPU: {cnt_gpu}")
        # for i in range(deviceCount):
        handle = nvmlDeviceGetHandleByIndex(0)

        while self.keep_monitor:
            power_gpu = (nvmlDeviceGetPowerUsage(handle) / 1000.0)
            lst_power_gpu.append(power_gpu)
            sleep(self.params["power"]["sampling_ms"] / 1000.0)
        nvmlShutdown()
        self.params["power"]["avg_watt_GPU"] = np.mean(lst_power_gpu)
        self.params["power"]["joules_GPU"] = self.params["power"]["avg_watt_GPU"] * self.params["time_total"]
        self.params["power"]["joules_total"] += self.params["power"]["joules_GPU"]
