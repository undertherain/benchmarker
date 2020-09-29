import threading
from time import sleep
import pyRAPL
import numpy as np
from py3nvml.py3nvml import nvmlInit, nvmlShutdown
from py3nvml.py3nvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetPowerUsage

class power_monitor_GPU:

    def __init__(self, params):
        # TODO: don't do this if GPU is not used
        nvmlInit()
        self.params = params
        self.keep_monitor = True

    def monitor(self):
        self.lst_power_gpu = []
        handles = [nvmlDeviceGetHandleByIndex(i) for i in self.params["gpus"]]
        while self.keep_monitor:
            power_gpu = [nvmlDeviceGetPowerUsage(handle) / 1000.0 for handle in handles]
            self.lst_power_gpu.append(sum(power_gpu))
            sleep(self.params["power"]["sampling_ms"] / 1000.0)

    def start(self):
        self.thread_monitor = threading.Thread(target=self.monitor, args=())
        self.thread_monitor.start()

    def stop(self):
        self.keep_monitor = False
        self.thread_monitor.join()
        nvmlShutdown()
        self.params["power"]["avg_watt_GPU"] = np.mean(self.lst_power_gpu)
        self.params["power"]["joules_GPU"] = self.params["power"]["avg_watt_GPU"] * self.params["time_total"]
        self.params["power"]["joules_total"] += self.params["power"]["joules_GPU"]
        self.params["power"]["avg_watt_total"] += self.params["power"]["avg_watt_GPU"]
        self.params["samples_per_joule_GPU"] = self.params["problem"]["cnt_samples"] * self.params["nb_epoch"] / self.params["power"]["joules_GPU"]


class power_monitor_RAPL:
    def __init__(self, params):
        self.params = params
        try:
            pyRAPL.setup()
            self.rapl_enabled = True
        except:
            self.rapl_enabled = False

    def start(self):
        if self.rapl_enabled:
            meter_rapl = pyRAPL.Measurement('bar')
            meter_rapl.begin()

    def stop(self):
        if self.rapl_enabled:
            self.meter_rapl.end()
            self.params["power"]["joules_CPU"] = sum(meter_rapl.result.pkg) / 1000000.0
            self.params["power"]["joules_RAM"] = sum(meter_rapl.result.dram) / 1000000.0
            self.params["power"]["avg_watt_CPU"] = self.params["power"]["joules_CPU"] / self.params["time_total"]
            self.params["power"]["avg_watt_RAM"] = self.params["power"]["joules_RAM"] / self.params["time_total"]
            self.params["power"]["joules_total"] += self.params["power"]["joules_CPU"]
            self.params["power"]["joules_total"] += self.params["power"]["joules_RAM"]
            self.params["power"]["avg_watt_total"] += self.params["power"]["avg_watt_CPU"]
            self.params["power"]["avg_watt_total"] += self.params["power"]["avg_watt_RAM"]
