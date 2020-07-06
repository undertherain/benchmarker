//#include <stdio.h>
//#include <stdlib.h>
#include <iostream>
#include <cblas.h>
#include <omp.h>
#include <chrono> 
#include "config.h"
#include "args.hpp"
#include <unistd.h>

using namespace std::chrono; 


int main(int argc, char * argv[]) {
    size_t m, n, k;
    float * A, *B, *C;
    double dtime;
    args_to_matrices(argc, argv, m, n, k, A, B, C);
    auto start = high_resolution_clock::now(); 
    // move to gpu
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, m * n * sizeof(float));
    cudaMalloc(&d_B, n * k * sizeof(float));
    cudaMalloc(&d_C, m * k * sizeof(float));
    cudaMemcpy(d_A, A, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n * k * sizeof(float), cudaMemcpyHostToDevice);
     
    // call cublas
    // sync
    // cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, m, k, n, 1, A, m, B, k, 1, C, m);
    sleep(1);
    auto stop = high_resolution_clock::now();
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
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
