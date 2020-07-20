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
    parse_args(argc, argv, precision, m, k, n);
    get_matrices<float>(m, k, n, A, B, C);
    const float alpha = 1;
    const float beta = 0;
    const size_t lda=m; // k for row major;
    const size_t ldb=k; //n; 
    const size_t ldc=m; //n;
    auto start = high_resolution_clock::now(); 
    if (precision == "FP32")
        cblas_sgemm(CblasColMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    m,
                    n,
                    k,
                    alpha,
                    A, lda,
                    B, ldb,
                    beta,
                    C, ldc);
    else
	{
        fprintf(stderr, "not implemented yet");
	    throw "madamada";
	}
    std::cerr << "MNK " << m << " " << n << " " << k << std::endl;
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
