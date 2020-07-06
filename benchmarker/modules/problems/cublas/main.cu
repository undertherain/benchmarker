#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <chrono>
#include "config.h"
#include "args.hpp"

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


int main(int argc, char * argv[]) {
    size_t m, n, k;
    float *A, *B, *C;
    double dtime;
    args_to_matrices<float>(argc, argv, m, n, k, A, B, C);
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, m * k * sizeof(float));
    cudaMalloc(&d_B, k * n * sizeof(float));
    cudaMalloc(&d_C, m * k * sizeof(float));
    cudaMemcpy(d_A, A, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n * k * sizeof(float), cudaMemcpyHostToDevice);
    cublasHandle_t handle;
    const float alf = 1;
    const float bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;
    int lda=m, ldb=k, ldc=m;
    int gpu_id = 0; // TODO: get from command line
    cudaSetDevice(gpu_id);
    cublasCreate(&handle);
    auto start = high_resolution_clock::now(); 
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc);
    cudaDeviceSynchronize();
    auto stop = high_resolution_clock::now();
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaGetLastError());
    std::chrono::duration<double> seconds = (stop - start); 
    dtime = seconds.count();
    double gflop = (2.0 * m * n * k) / (1024 * 1024 * 1024);
    double gflops = gflop / dtime;
    printf("gflops: \t%f\n", gflop);
    printf("time: \t%f\n", dtime);
    printf("ips: \t%f\n", 1 / dtime);
    printf("gflops/s: \t%f\n", gflops);
    return 0;
}
