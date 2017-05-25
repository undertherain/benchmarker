#include <omp.h>
#include <stdlib.h>
#include <stdio.h>

typedef float t_float;
#define ALIGNMENT 1024*4

void gemm(t_float * restrict _A, t_float * restrict _B , t_float * restrict _C, size_t n) 
{     
//	A = __builtin_assume_aligned (A, ALIGNMENT);
  	//B = __builtin_assume_aligned (B, ALIGNMENT);
  	//C = __builtin_assume_aligned (C, ALIGNMENT);
  	t_float *A = __builtin_assume_aligned(_A, ALIGNMENT);
  	t_float *B = __builtin_assume_aligned(_B, ALIGNMENT);
  	t_float *C = __builtin_assume_aligned(_C, ALIGNMENT);


   	for (size_t i = 0; i < n; i ++) 
		for (size_t j = 0; j < n; j++) 
		{
			t_float dot = 0;
			//#pragma omp simd
			//#pragma GCC ivdep
        	for (size_t k = 0; k < n; k++) 
        	{
          		dot +=  A[i*n+k] * B[k*n+j];
        	}
        	C[i*n+j] = dot;
		} 
}

int main(int argc, char * argv[])
{
	double dtime;
	size_t n = 1024;

	t_float * A = (t_float *) malloc (n * n * sizeof(t_float));
	t_float * B = (t_float *) malloc (n * n * sizeof(t_float));
	t_float * C = (t_float *) malloc (n * n * sizeof(t_float));

    for(size_t i=0; i<n*n; i++) { A[i] = rand()/RAND_MAX; B[i] = rand()/RAND_MAX; C[i] = 0;}

	dtime = omp_get_wtime();

   	for (size_t i = 0; i < n; i ++) 
		for (size_t j = 0; j < n; j++) 
		{
			t_float dot = 0;
        	for (size_t k = 0; k < n; k++) 
        	{
          		dot +=  A[i*n+k] * B[k*n+j];
        	}
        	C[i*n+j] = dot;
		} 

    dtime = omp_get_wtime() - dtime;
    printf("in main     \t%f\n", dtime);

	dtime = omp_get_wtime();
	gemm(A,B,C,n);
    dtime = omp_get_wtime() - dtime;
    printf("in function\t%f\n", dtime);

	free (A);
	free (B);
	free (C);
}
