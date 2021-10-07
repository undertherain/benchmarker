from .i_gemm import IGEMM


class Benchmark(IGEMM):
    def expected_precisions(self):
        return ["FP32", "FP16", "mixed"]

    def expect_gpus(self):
        return True
