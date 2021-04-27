import threading
from time import sleep

import numpy as np


class PowerMonitorNVML:
    def __init__(self, params):
        import py3nvml.py3nvml as nvml

        self.nvml = nvml
        # from py3nvml.py3nvml import nvmlInit, nvmlShutdown
        # from py3nvml.py3nvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetPowerUsage
        nvml.nvmlInit()
        self.params = params
        self.keep_monitor = True
        # def start(self):
        self.thread_monitor = threading.Thread(target=self.monitor, args=())
        self.thread_monitor.start()

    def monitor(self):
        self.lst_power_gpu = []
        handles = [self.nvml.nvmlDeviceGetHandleByIndex(i) for i in self.params["gpus"]]
        while self.keep_monitor:
            power_gpu = [self.nvml.nvmlDeviceGetPowerUsage(h) / 1000.0 for h in handles]
            self.lst_power_gpu.append(sum(power_gpu))
            sleep(self.params["power"]["sampling_ms"] / 1000.0)

    def stop(self):
        self.keep_monitor = False
        self.thread_monitor.join()
        self.nvml.nvmlShutdown()
        # a small hack to remove pre-heat time measurement
        self.post_process()

    def post_process(self):
        percent_measurement_remove = 0.1
        cnt_measurements_remove = int(
            len(self.lst_power_gpu) * percent_measurement_remove
        )
        measurements_trimmed = self.lst_power_gpu[cnt_measurements_remove:]
        self.params["power"]["avg_watt_GPU"] = np.mean(measurements_trimmed)
        # self.params["power"]["joules_GPU"] = self.params["power"]["avg_watt_GPU"] * self.params["time_total"]
        # self.params["power"]["joules_total"] += self.params["power"]["joules_GPU"]
        # self.params["power"]["avg_watt_total"] += self.params["power"]["avg_watt_GPU"]
        # if "cnt_samples" in self.params["problem"]:
        #     self.params["samples_per_joule_GPU"] = self.params["problem"]["cnt_samples"] * self.params["nb_epoch"] / self.params["power"]["joules_GPU"]
