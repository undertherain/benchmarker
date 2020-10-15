import os
from .i_gemm import IGEMM
from benchmarker.util.abstractprocess import Process
from .power_mon import power_monitor_GPU


class Benchmark(IGEMM):
    def __init__(self, params, remaining_args=None):
        super().__init__(params, remaining_args)
        assert self.params["problem"]["precision"] in ["FP32", "FP16", "mixed"]

    def run(self):
        if "nb_gpus" in self.params:
            if self.params["nb_gpus"] != 1:
                raise Exception("cublas requires one GPU")
        #size = " ".join(map(str, self.params['problem']['size']))
        path_binary = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                   "problems/gemm/cublas/main")
        if not os.path.isfile(path_binary):
            raise(RuntimeError(f"{path_binary} not found, run make manually"))
        command = [path_binary,
                   self.params["problem"]["precision"],
                   * map(str, self.params['problem']['size']),
                   str(self.params["nb_epoch"])]
        # TODO(Alex): think of how to reuse this across all problems
        power_monitor_gpu = power_monitor_GPU(self.params)
        power_monitor_gpu.start()
        process = Process(command=command)
        result = process.get_output()
        std_out = result["out"]
        elapsed_time = float(std_out.strip())
        self.params["time_total"] = elapsed_time
        power_monitor_gpu.stop()
        self.params["time_epoch"] = elapsed_time / self.params["nb_epoch"]
        # TODO: unify flops per sec name with neural nets
        self.params["GFLOP/sec"] = self.params["GFLOP"] / elapsed_time
        self.params["gops_per_joule"] = self.params["GFLOP"] / self.params["power"]["joules_total"]
