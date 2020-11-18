import importlib
from .power_mon import power_monitor_GPU, power_monitor_RAPL


class IBenchmark:
    def measure_power_and_run(self):
        if self.params["nb_gpus"] > 0:
            power_monitor_gpu = power_monitor_GPU(self.params)
            power_monitor_gpu.start()
        power_monitor_cpu = power_monitor_RAPL(self.params)
        power_monitor_cpu.start()
        try:
            results = self.run()
        finally:
            if self.params["nb_gpus"] > 0:
                power_monitor_gpu.stop()
        if self.params["nb_gpus"] > 0:
            power_monitor_gpu.post_process()
        power_monitor_cpu.stop()
        self.post_process()
        return results

    def load_data(self):
        params = self.params
        mod = importlib.import_module(
            "benchmarker.modules.problems." + params["problem"]["name"] + ".data"
        )
        get_data = getattr(mod, "get_data")
        data = get_data(params)
        return data
