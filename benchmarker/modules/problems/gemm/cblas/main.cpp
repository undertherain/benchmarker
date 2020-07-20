#include <iostream>
#include <chrono>
#include <cblas.h>
#include "../args.hpp"

using namespace std::chrono; 



int main(int argc, char * argv[]) {
    size_t m, n, k;
    float *A, *B, *C;
    double dtime;
    std::string precision(argv[1]);
    args_to_matrices<float>(argc - 1, argv + 1, m, n, k, A, B, C);
    const float alpha = 1;
    const float beta = 0;
    // int lda=m, ldb=k, ldc=m;
    auto start = high_resolution_clock::now(); 
    // TODO: this m n k ordering is a mess, rename them intuitively ><
    if (precision == "FP32")
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, m, k, n, alpha, A, m, B, k, beta, C, m);
        //cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc);
    else
        throw "madamada";
	//cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, 
        //             alpha, d_A, CUDA_R_16F, lda, d_B, CUDA_R_16F, ldb, beta, d_C, CUDA_R_32F, ldc, CUDA_R_32F,
        //             CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    auto stop = high_resolution_clock::now();
    std::chrono::duration<double> seconds = (stop - start); 
    dtime = seconds.count();
    double gflop = (2.0 * m * n * k) / (1024 * 1024 * 1024);
    double gflops = gflop / dtime;
    printf("%f\n", dtime);
    fprintf(stderr, "gflops: \t%f\n", gflop);
    fprintf(stderr, "time: \t%f\n", dtime);
    fprintf(stderr, "ips: \t%f\n", 1 / dtime);
    fprintf(stderr, "gflops/s: \t%f\n", gflops);
    return 0;
}
