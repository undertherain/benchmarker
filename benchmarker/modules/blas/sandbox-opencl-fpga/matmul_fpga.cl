
/*
 * Baseline "naive" version
 * Artur Podobas
 */
#include "ihc_apint.h"

__kernel void matmul(__global float * __restrict A, __global float * __restrict B, __global float * __restrict C, unsigned int n)
{
  for (size_t i = 0; i < n; i ++) 
   for (size_t j = 0; j < n; j++) 
    {
       float dot = 0;              
       for (size_t k = 0; k < n; k++) 
          dot +=  A[i*n+k] * B[k*n+j];
       C[i*n+j] = dot;
    } 
}

