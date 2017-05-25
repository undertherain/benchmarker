#include <stdarg.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cstring>
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"

using namespace aocl_utils;

static cl_platform_id platform = NULL;
static cl_device_id device = NULL;
static cl_context context = NULL;
static cl_command_queue queue = NULL;
static cl_kernel kernel = NULL;
static cl_program program = NULL;



/*****************************************************************
 *                      Helper functions                         *
 *****************************************************************/
cl_mem CL_Malloc ( size_t size, cl_mem_flags flags)
{
   cl_int ret;
   cl_mem rt = clCreateBuffer(context, flags, size, NULL, &ret);
   if (ret != CL_SUCCESS) 
	{ fprintf(stderr,"Couldnt not allocate\n");exit(0);}
   return rt;   
}

void memcpy2dev ( cl_mem dst, void *src, size_t size)
{
  clEnqueueWriteBuffer (queue, dst, CL_TRUE, 0 , size , src , 0 , NULL , NULL);
}

void memcpy2host ( void *dst, cl_mem src, size_t size)
{
  clEnqueueReadBuffer(queue, src, CL_TRUE, 0, size, dst, 0, NULL, NULL); 
}


void gemm(unsigned int n) 
{     
	float * A = (float *) malloc (n * n * sizeof(float));
	float * B = (float *) malloc (n * n * sizeof(float));
	float * C = (float *) malloc (n * n * sizeof(float));
	float * C_host_fpga = (float *) malloc (n * n * sizeof(float));

	/* Initialize the arrays */
	for(size_t i=0; i<n*n; i++) 
	   { A[i] = ((float) (rand() % 0xdead) / (float) 0xbeef); B[i] = ((float) (rand() % 0xdead) / (float) 0xbeef); C[i] = 0; C_host_fpga[i] = 0;}

	/* Run it serially for testing */
	for (size_t i = 0; i < n; i ++) 
	  for (size_t j = 0; j < n; j++) 
	   {
	     float dot = 0;
             for (size_t k = 0; k < n; k++) 
          		dot +=  A[i*n+k] * B[k*n+j];
        	C[i*n+j] = dot;
   	   } 

	
	/* Allocate host-side memory */
	cl_mem A_fpga = CL_Malloc(n*n*sizeof(float) , CL_MEM_READ_WRITE);
	cl_mem B_fpga = CL_Malloc(n*n*sizeof(float) , CL_MEM_READ_WRITE);
	cl_mem C_fpga = CL_Malloc(n*n*sizeof(float) , CL_MEM_READ_WRITE);
	memcpy2dev(A_fpga, A, n * n * sizeof(float));
	memcpy2dev(B_fpga, B, n * n * sizeof(float));


	/* Set program arguments */
	cl_int ret;
    if ( (ret = clSetKernelArg(kernel, 0, sizeof(float *), (void *) &A_fpga)) != CL_SUCCESS) 
	{ fprintf(stderr,"Error in launching kernel.\n"); exit(0); }
    if ( (ret = clSetKernelArg(kernel, 1, sizeof(float *), (void *) &B_fpga)) != CL_SUCCESS) 
	{ fprintf(stderr,"Error in launching kernel.\n"); exit(0); }
    if ( (ret = clSetKernelArg(kernel, 2, sizeof(float *), (void *) &C_fpga)) != CL_SUCCESS) 
	{ fprintf(stderr,"Error in launching kernel.\n"); exit(0); }
    if ( (ret = clSetKernelArg(kernel, 3, sizeof(unsigned int), (void *) &n)) != CL_SUCCESS) 
	{ fprintf(stderr,"Error in launching kernel.\n"); exit(0); }
	
	/* Start the kernel */
    if ( (ret = clEnqueueTask(queue, kernel, 0, NULL,NULL)) != CL_SUCCESS)
	{ fprintf(stderr,"Error in launching kernel.\n"); exit(0); }
	
	/* Wait 'til the queue is empty */
    clFinish(queue);

    /* Copy the result back from FPGA */
    memcpy2host ( C_host_fpga, C_fpga,  n * n * sizeof(float));

    /* Verify correctness */
    for (size_t i = 0; i < n*n; i ++) 
	if ( fabs(C_host_fpga[i] - C[i]) > 0.0001)
	{ fprintf(stderr,"Error result: %f vs %f\n", C_host_fpga[i], C[i]); exit(0); }

    fprintf(stderr,"Correct results.\n");
}

/* Called from checkError() */
void cleanup() {
  if(kernel) clReleaseKernel(kernel);if (program) clReleaseProgram(program); if (queue) clReleaseCommandQueue(queue); if (context) clReleaseContext(context);
}


/****************************************************************/
// Entry point.

int main ( int argc, char *argv[])
{
  cl_int status;
  if(!setCwdToExeDir()) 
    exit(0);
  if ( (platform = findPlatform("Intel(R) FPGA")) == NULL)
    { fprintf(stderr,"Cannot find OpenCL platform\n");exit(0);}
  /* Find first device, loading binaries, find kernel etc. */
  scoped_array<cl_device_id> devices;
  cl_uint num_devices;
  devices.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));
  device = devices[0];
  context = clCreateContext(NULL, 1, &device, &oclContextCallback, NULL, &status);
  checkError(status, "Error in clCreateContext");
  queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Error in clCreateCommandQueue");
  std::string binary_file = getBoardBinaryFile("matmul_fpga", device); 
  program = createProgramFromBinary(context, binary_file.c_str(), &device, 1);
  status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
  checkError(status, "Failed to build matmul");
  const char *kernel_name = "matmul"; 
  kernel = clCreateKernel(program, kernel_name, &status);
  checkError(status, "Failed to create matmul kernel");


  fprintf(stderr,"\n\n----------------------------\n-     Starting MatMul Test -\n----------------------------\n\n");
  gemm(256); 

}

