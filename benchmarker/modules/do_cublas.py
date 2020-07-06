from timeit import default_timer as timer
import numpy as np
from .i_gemm import IGEMM
from benchmarker.util.abstractprocess import Process


class Benchmark(IGEMM):
    def __init__(self, params, remaining_args=None):
        super().__init__(params, remaining_args)

    def run(self):
        if "nb_gpus" in self.params:
            if self.params["nb_gpus"] != 1:
                raise Exception("cublas requires one GPU")
        M, N, K = self.matrix_size
        dtype = np.float32
        a = np.random.random((M, N)).astype(dtype)
        b = np.random.random((N, K)).astype(dtype)
        c = np.random.random((M, K)).astype(dtype)
        nb_epoch = 2
        # print(self.params['problem']['size'])
        size = " ".join(map(str, self.params['problem']['size']))
        command = ["/home/blackbird/Projects_heavy/performance/benchmarker/benchmarker/modules/problems/cublas/main",
                   self.params["problem"]["precision"],
                   size]
        # print(command)
        process = Process(command=command)
        result = process.get_output()
        std_out = result["out"]
        # print(std_out)
        elapsed_time = float(std_out.strip())
        self.params["time"] = elapsed_time
        self.params["GFLOP/sec"] = self.params["GFLOP"] / elapsed_time
