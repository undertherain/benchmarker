#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "config.h"

void transpose(t_float *A, t_float *B, int n) {
    int i,j;
    for(i=0; i<n; i++) {
        for(j=0; j<n; j++) {
            B[j*n+i] = A[i*n+j];
        }
    }
}

void gemm(t_float *A, t_float *B, t_float *C, int n) 
{   
    int i, j, k;
    for (i = 0; i < n; i++) { 
        for (j = 0; j < n; j++) {
            t_float dot  = 0;
            for (k = 0; k < n; k++) {
                dot += A[i*n+k]*B[k*n+j];
            } 
            C[i*n+j ] = dot;
        }
    }
}

void gemm_omp(t_float *A, t_float *B, t_float *C, int n) 
{   
    #pragma omp parallel
    {
        int i, j, k;
        #pragma omp for
        for (i = 0; i < n; i++) { 
            for (j = 0; j < n; j++) {
                t_float dot  = 0;
                for (k = 0; k < n; k++) {
                    dot += A[i*n+k]*B[k*n+j];
                } 
                C[i*n+j ] = dot;
            }
        }

    }
}

void gemmT(t_float *A, t_float *B, t_float *C, int n) 
{   
    int i, j, k;
    t_float *B2;
    B2 = (t_float*)malloc(sizeof(t_float)*n*n);
    transpose(B,B2, n);
    for (i = 0; i < n; i++) { 
        for (j = 0; j < n; j++) {
            t_float dot  = 0;
            for (k = 0; k < n; k++) {
                dot += A[i*n+k]*B2[j*n+k];
            } 
            C[i*n+j ] = dot;
        }
    }
    free(B2);
}

void gemmT_omp(t_float *A, t_float *B, t_float *C, int n) 
{   
    t_float *B2;
    B2 = (t_float*)malloc(sizeof(t_float)*n*n);
    transpose(B,B2, n);
    #pragma omp parallel
    {
        int i, j, k;
        #pragma omp for
        for (i = 0; i < n; i++) { 
            for (j = 0; j < n; j++) {
                t_float dot  = 0;
                for (k = 0; k < n; k++) {
                    dot += A[i*n+k]*B2[j*n+k];
                } 
                C[i*n+j ] = dot;
            }
        }

    }
    free(B2);
}

int main() {
    size_t i;
    t_float *A, *B, *C;
    double dtime;

    A = (t_float*) malloc(sizeof(t_float)*n*n);
    B = (t_float*) malloc(sizeof(t_float)*n*n);
    C = (t_float*) malloc(sizeof(t_float)*n*n);
    for(i=0; i<n*n; i++) { A[i] = rand()/RAND_MAX; B[i] = rand()/RAND_MAX;}

    dtime = omp_get_wtime();
    gemm(A,B,C, n);
    dtime = omp_get_wtime() - dtime;
    printf("no transpose  no openmp\t%f\n", dtime);

    dtime = omp_get_wtime();
    gemm_omp(A,B,C, n);
    dtime = omp_get_wtime() - dtime;
    printf("no transpose, openmp\t%f\n", dtime);

    dtime = omp_get_wtime();
    gemmT(A,B,C, n);
    dtime = omp_get_wtime() - dtime;
    printf("tranpose, no openmp\t%f\n", dtime);

    dtime = omp_get_wtime();
    gemmT_omp(A,B,C, n);
    dtime = omp_get_wtime() - dtime;
    printf("transpose, openmp\t%f\n", dtime);



    return 0;

}