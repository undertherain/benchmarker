#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
#include <omp.h>
#include "config.h"
#include "args.hpp"


int main(int argc, char * argv[]) {
    // this is super-upgly, but maybe I'll fix it some time later :)
    fprintf(stderr, "doing clbas\n");
    size_t m, n, k;
    float * A, *B, *C;
    double dtime;
    args_to_matrices(argc, argv, m, n, k, A, B, C);
    dtime = omp_get_wtime();
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, m, k, n, 1, A, m, B, k, 1, C, m);
    dtime = omp_get_wtime() - dtime;
    double gflop = (2.0 * m * n * k) / (1024 * 1024 * 1024);
    double gflops = gflop / dtime;
    printf("gflops: \t%f\n", gflop);
    printf("time: \t%f\n", dtime);
    printf("ips: \t%f\n", 1 / dtime);
    printf("gflops/s: \t%f\n", gflops);
    return 0;
}
