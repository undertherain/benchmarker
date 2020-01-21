#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
#include <omp.h>
#include "config.h"

int main(int argc, char * argv[]) {
    long m, n, k;
    if (argc==2) {
        m = atoi(argv[1]);
        n = m;
        k = m;
    }
    else
        if (argc==4) {
            m = atoi(argv[1]);
            n = atoi(argv[2]);
            k = atoi(argv[3]);
        }
        else
            return -1;
    t_float *A, *B, *C;
    double dtime;
    size_t i;
    // TODO: move this to shared module
    fprintf(stderr, "doing clbas\n");
    A = (t_float*) malloc(sizeof(t_float)*m*n);
    B = (t_float*) malloc(sizeof(t_float)*n*k);
    C = (t_float*) malloc(sizeof(t_float)*m*k);
    for(i=0; i<n*n; i++) { A[i] = rand()/RAND_MAX; B[i] = rand()/RAND_MAX;}
    fprintf(stderr, "done random init\n");

    dtime = omp_get_wtime();
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,n,n,n, 1, A, n, B, n, 1, C ,n);
    dtime = omp_get_wtime() - dtime;
    double gflop = (2.0 * n * n * n) / (1024 * 1024 * 1024);
    double gflops = gflop / dtime;
    printf("time: \t%f\n", dtime);
    printf("gflops/s: \t%f\n", gflops);
}
