import os
from .i_gemm import IGEMM
from benchmarker.util import get_script_dir
from benchmarker.util.abstractprocess import Process


class Benchmark(IGEMM):
    def __init__(self, params, remaining_args=None):
        super().__init__(params, remaining_args)

    def run(self):
        if "nb_gpus" in self.params:
            if self.params["nb_gpus"] != 1:
                raise Exception("cublas requires one GPU")
        size = " ".join(map(str, self.params['problem']['size']))
        command = [os.path.join(get_script_dir(), "../modules/problems/cublas/main"),
                   self.params["problem"]["precision"],
                   size]
        process = Process(command=command)
        result = process.get_output()
        std_out = result["out"]
        elapsed_time = float(std_out.strip())
        self.params["time"] = elapsed_time
        self.params["GFLOP/sec"] = self.params["GFLOP"] / elapsed_time
