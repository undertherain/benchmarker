#include <iostream>
#include <chrono>
#include <cblas.h>
#include "../args.hpp"

using namespace std::chrono; 



int main(int argc, char * argv[]) {
    size_t m, n, k;
    float *A, *B, *C;
    double dtime;
    std::string precision;
    parse_args(argc, argv, precision, m, n, k);
    get_matrices<float>(m, n, k, A, B, C);
    const float alpha = 1;
    const float beta = 0;
    // int lda=m, ldb=k, ldc=m;
    auto start = high_resolution_clock::now(); 
    // TODO: this m n k ordering is a mess, rename them intuitively ><
    if (precision == "FP32")
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, m, k, n, alpha, A, m, B, k, beta, C, m);
    else
	{
        fprintf(stderr, "not implemented yet");
	    throw "madamada";
	}
    std::cerr<<"MNK " << m << " " << n << " " << k << std::endl;
    auto stop = high_resolution_clock::now();
    std::chrono::duration<double> seconds = (stop - start); 
    dtime = seconds.count();
    double gflop = (2.0 * m * n * k) / (1024 * 1024 * 1024);
    double gflops = gflop / dtime;
    printf("%f\n", dtime);
    fprintf(stderr, "gflops: \t%f\n", gflop);
    fprintf(stderr, "time: \t%f\n", dtime);
    fprintf(stderr, "ips: \t%f\n", 1 / dtime);
    fprintf(stderr, "gflops/s: \t%f\n", gflops);
    return 0;
}
