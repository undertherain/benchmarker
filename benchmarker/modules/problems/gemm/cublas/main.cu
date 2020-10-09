#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <chrono>
#include "../args.hpp"

using namespace std::chrono; 

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)
template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}


template<typename type_numerics>
void call_blas(cublasHandle_t handle, 
                      int m,
                      int n,
                      int k,
                      const type_numerics * alpha, 
                      type_numerics * d_A, 
                      int lda, 
                      type_numerics * d_B, 
                      int ldb, 
                      const type_numerics * beta,
                      type_numerics * d_C, 
                      int ldc,
                      std::string precision);

template<>
void call_blas<float>(cublasHandle_t handle, 
                      int m,
                      int n,
                      int k,
                      const float * alpha, 
                      float * d_A, 
                      int lda, 
                      float * d_B, 
                      int ldb, 
                      const float * beta, 
                      float * d_C, 
                      int ldc,
                      std::string precision)
{
    if (precision == "FP32")
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                     alpha, d_A, lda, d_B, ldb, beta, d_C, ldc);
    if (precision == "mixed")
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, 
                     alpha, d_A, CUDA_R_16F, lda, d_B, CUDA_R_16F, ldb, beta, d_C, CUDA_R_32F, ldc, CUDA_R_32F,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

template<>
void call_blas<__half>(cublasHandle_t handle, 
                      int m,
                      int n,
                      int k,
                      const __half * alpha, 
                      __half * d_A, 
                      int lda, 
                      __half * d_B, 
                      int ldb, 
                      const __half * beta, 
                      __half * d_C, 
                      int ldc,
                      std::string precision)
{
    cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                alpha, d_A, lda, d_B, ldb, beta, d_C, ldc);
}


template<typename type_numerics>
double call_blas_and_measure_seconds(
    size_t m, 
    size_t n,
    size_t k,
    std::string precision) 
{
    int nb_epoch = 10
    type_numerics *A, *B, *C;
    type_numerics *d_A, *d_B, *d_C;
    get_matrices<type_numerics>(m, k, n, A, B, C);
    std::cerr << "done init get_matrices\n";
    cudaMalloc(&d_A, m * k * sizeof(type_numerics));
    cudaMalloc(&d_B, k * n * sizeof(type_numerics));
    cudaMalloc(&d_C, m * n * sizeof(type_numerics));
    cudaMemcpy(d_A, A, m * k * sizeof(type_numerics), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, k * n * sizeof(type_numerics), cudaMemcpyHostToDevice);
    cublasHandle_t handle;
    const type_numerics alf = 1;
    const type_numerics bet = 0;
    const type_numerics *alpha = &alf;
    const type_numerics *beta = &bet;
    int lda=m, ldb=k, ldc=m;
    int gpu_id = 0; // this is actually OK if calle from Benchmarker bec. visible devices
    cudaSetDevice(gpu_id);
    cublasCreate(&handle);
    auto start = high_resolution_clock::now(); 
    // cublas only does column-major order
    for(size_t i=0; i<nb_epoch; i++)
      call_blas<type_numerics> (handle, m, n, k,
                                alpha, d_A, lda, 
                                d_B, ldb, beta, 
                                d_C, ldc, precision);

    cudaDeviceSynchronize();
    auto stop = high_resolution_clock::now();
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
    checkCudaErrors(cudaGetLastError());
    std::chrono::duration<double> seconds = (stop - start); 
    return seconds.count() / nb_epoch;
}

int main(int argc, char * argv[]) {
    size_t m, n, k;
    double dtime;
    std::string precision;
    parse_args(argc, argv, precision, m, k, n);
    std::cerr << "done parsing args\n";
    if (precision == "FP32")
        dtime = call_blas_and_measure_seconds<float>(m, n, k, precision);
    if (precision == "FP16")
        dtime = call_blas_and_measure_seconds<__half>(m, n, k, precision);
    if (precision == "mixed")
        dtime = call_blas_and_measure_seconds<float>(m, n, k, precision);

    double gflop = (2.0 * m * n * k) / (1024 * 1024 * 1024);
    double gflops = gflop / dtime;
    printf("%f\n", dtime);
    fprintf(stderr, "gflops: \t%f\n", gflop);
    fprintf(stderr, "time: \t%f\n", dtime);
    fprintf(stderr, "ips: \t%f\n", 1 / dtime);
    fprintf(stderr, "gflops/s: \t%f\n", gflops);
    return 0;
}
