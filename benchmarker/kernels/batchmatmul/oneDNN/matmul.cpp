#include <iostream>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include "oneapi/dnnl/dnnl.hpp"
#include "example_utils.hpp"

using namespace std::chrono; 
using namespace dnnl;
using tag = memory::format_tag;
using dt = memory::data_type;

int main(int argc, char * argv[]) {
    if (argc != 7)
    {
        std::cerr << "provide precision, m, n, k, batch_size, nb_epoch as command line parameters\n";
        std::cerr << "got " << argc << " parameters\n";
        exit(-1);
    }
    std::string precision = std::string(argv[1]);
    if(precision != "FP32"){
        std::cerr << "sorry, only support FP32 right now.\n";
    }
    double dtime;
    // Create execution dnnl::engine.
    dnnl::engine engine(dnnl::engine::kind::cpu, 0);
    // Create dnnl::stream.
    dnnl::stream engine_stream(engine);
    // Tensor dimensions.
    const memory::dim MB = atoi(argv[5]), // batch size
            M = atoi(argv[2]), K = atoi(argv[4]), N = atoi(argv[3]);
    size_t nb_epoch = atoi(argv[6]);

    // Source (src), weights, bias, and destination (dst) tensors dimensions.
    memory::dims src_dims = {MB, M, K};
    memory::dims weights_dims = {MB, K, N};
    memory::dims bias_dims = {1, 1, N};
    memory::dims dst_dims = {MB, M, N};

    // Allocate buffers.
    std::vector<float> src_data(product(src_dims));
    std::vector<float> weights_data(product(weights_dims));
    std::vector<float> bias_data(product(bias_dims));
    std::vector<float> dst_data(product(dst_dims));
    // Initialize src, weights, bias.
    std::generate(src_data.begin(), src_data.end(), []() {
        return rand()/RAND_MAX;
    });
    std::generate(weights_data.begin(), weights_data.end(), []() {
        return rand()/RAND_MAX;
    });
    std::generate(bias_data.begin(), bias_data.end(), []() {
        return rand()/RAND_MAX;
    });
    // Create memory descriptors and memory objects for src, weights, bias, and dst.
    auto src_md = memory::desc(src_dims, dt::f32, tag::abc);
    auto weights_md = memory::desc(weights_dims, dt::f32, tag::abc);
    auto bias_md = memory::desc(bias_dims, dt::f32, tag::abc);
    auto dst_md = memory::desc(dst_dims, dt::f32, tag::abc);
    auto src_mem = memory(src_md, engine);
    auto weights_mem = memory(weights_md, engine);
    auto bias_mem = memory(bias_md, engine);
    auto dst_mem = memory(dst_md, engine);

    // Write data to memory object's handles.
    write_to_dnnl_memory(src_data.data(), src_mem);
    write_to_dnnl_memory(weights_data.data(), weights_mem);
    write_to_dnnl_memory(bias_data.data(), bias_mem);

    // Create operation descriptor
    auto matmul_d = matmul::desc(src_md, weights_md, bias_md, dst_md);

    // Create primitive descriptor.
    auto matmul_pd = matmul::primitive_desc(matmul_d, /*matmul_attr,*/ engine);
    
    // Create the primitive.
    auto matmul_prim = matmul(matmul_pd);
    
    // Primitive arguments.
    std::unordered_map<int, memory> matmul_args;
    matmul_args.insert({DNNL_ARG_SRC, src_mem});
    matmul_args.insert({DNNL_ARG_WEIGHTS, weights_mem});
    matmul_args.insert({DNNL_ARG_BIAS, bias_mem});
    matmul_args.insert({DNNL_ARG_DST, dst_mem});
    
    // Primitive execution: matrix multiplication with ReLU.
    auto start = high_resolution_clock::now();
    for(int i=0;i<nb_epoch;i++)
        matmul_prim.execute(engine_stream, matmul_args);
    
    // Wait for the computation to finalize.
    engine_stream.wait();
    
    std::cerr << "MNK " << M << " " << N << " " << K << std::endl;
    auto stop = high_resolution_clock::now();
    std::chrono::duration<double> seconds = (stop - start); 
    dtime = seconds.count();
    double gflop = (2.0 * M * N * K * MB) / (1000 * 1000 * 1000);
    gflop *= static_cast<double>(nb_epoch);
    double gflops = gflop / dtime;
    printf("%f\n", dtime);
    fprintf(stderr, "gflops: \t%f\n", gflop);
    fprintf(stderr, "time: \t%f\n", dtime);
    fprintf(stderr, "ips: \t%f\n", 1 / dtime);
    fprintf(stderr, "gflops/s: \t%f\n", gflops);
    return 0;
}
