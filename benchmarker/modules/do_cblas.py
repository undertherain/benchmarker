import os
from .i_gemm import IGEMM
from benchmarker.util.abstractprocess import Process


class Benchmark(IGEMM):
    def __init__(self, params, remaining_args=None):
        super().__init__(params, remaining_args)
        assert self.params["problem"]["precision"] in ["FP32", "mixed"]

    def run(self):
        if "nb_gpus" in self.params:
            if self.params["nb_gpus"] > 0:
                raise Exception("cblas does not work on GPU")
        size = " ".join(map(str, self.params['problem']['size']))
        # TODO: this does not work inless the binaries are copied to site_packages
        path_binary = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                   "problems/gemm/cblas/main")
        if not os.path.isfile(path_binary):
            raise(RuntimeError(f"{path_binary} not found, run make manually"))
        command = [path_binary,
                   self.params["problem"]["precision"],
                   size]
        process = Process(command=command)
        result = process.get_output()
        std_out = result["out"]
        elapsed_time = float(std_out.strip())
        self.params["time"] = elapsed_time
        self.params["GFLOP/sec"] = self.params["GFLOP"] / elapsed_time
