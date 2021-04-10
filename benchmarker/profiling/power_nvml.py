import threading
from time import sleep
import numpy as np

class power_monitor_GPU:

    def __init__(self, params):
        import py3nvml.py3nvml as nvml
        self.nvml = nvml
        # from py3nvml.py3nvml import nvmlInit, nvmlShutdown
        # from py3nvml.py3nvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetPowerUsage
        nvml.nvmlInit()
        self.params = params
        self.keep_monitor = True

    def monitor(self):
        self.lst_power_gpu = []
        handles = [self.nvml.nvmlDeviceGetHandleByIndex(i) for i in self.params["gpus"]]
        while self.keep_monitor:
            power_gpu = [self.nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0 for handle in handles]
            self.lst_power_gpu.append(sum(power_gpu))
            sleep(self.params["power"]["sampling_ms"] / 1000.0)

    def start(self):
        self.thread_monitor = threading.Thread(target=self.monitor, args=())
        self.thread_monitor.start()

    def stop(self):
        self.keep_monitor = False
        self.thread_monitor.join()
        self.nvml.nvmlShutdown()
        # a small hack to remove pre-heat time measurement

    def post_process(self):
        cnt_cut_measurements = min(len(self.lst_power_gpu), 100)
        measurements_trimmed = self.lst_power_gpu[:cnt_cut_measurements]
        self.params["power"]["avg_watt_GPU"] = np.mean(measurements_trimmed)
        self.params["power"]["joules_GPU"] = self.params["power"]["avg_watt_GPU"] * self.params["time_total"]
        self.params["power"]["joules_total"] += self.params["power"]["joules_GPU"]
        self.params["power"]["avg_watt_total"] += self.params["power"]["avg_watt_GPU"]
        if "cnt_samples" in self.params["problem"]:
            self.params["samples_per_joule_GPU"] = self.params["problem"]["cnt_samples"] * self.params["nb_epoch"] / self.params["power"]["joules_GPU"]



