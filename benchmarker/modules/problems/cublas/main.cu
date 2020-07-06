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
    // this is super-upgly, but maybe I'll fix it some time later :)
    // fprintf(stderr, "doing clbas\n");
    size_t m, n, k;
    float * A, *B, *C;
    double dtime;
    args_to_matrices(argc, argv, m, n, k, A, B, C);
    auto start = high_resolution_clock::now(); 
    // cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, m, k, n, 1, A, m, B, k, 1, C, m);
    sleep(1);
    auto stop = high_resolution_clock::now();
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
