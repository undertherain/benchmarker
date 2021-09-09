from .i_gemm import IGEMM


class Benchmark(IGEMM):
    def expect_gpus(self):
        return False
