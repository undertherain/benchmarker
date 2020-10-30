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
        except Exception as e:
            power_monitor_gpu.stop()
            raise e
        if self.params["nb_gpus"] > 0:
            power_monitor_gpu.stop()
            power_monitor_gpu.post_process()
        power_monitor_cpu.stop()
        self.post_process()
        return results
