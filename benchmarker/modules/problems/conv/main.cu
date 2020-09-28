// http://www.goldsborough.me/cuda/ml/cudnn/c++/2017/10/01/14-37-23-convolutions_with_cudnn/
//
// Download input image from the website.
//
// The docs here
// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetFilterNdDescriptor
// say that: "The tensor format CUDNN_TENSOR_NHWC has limited support
// in cudnnConvolutionForward(), cudnnConvolutionBackwardData(), and
// cudnnConvolutionBackwardFilter()." so for now let's stick to NCHW!

#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cudnn.h>
#include <cudnn_cnn_infer.h>
#include <cudnn_ops_infer.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>

#define checkCUDNN(expression)                                                 \
  {                                                                            \
    cudnnStatus_t status = (expression);                                       \
    if (status != CUDNN_STATUS_SUCCESS) {                                      \
      std::cerr << "Error on line " << __LINE__ << ": "                        \
                << cudnnGetErrorString(status) << std::endl;                   \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  }

const int MAX_DIM = 8;
class Args {
public:
  Args(const int argc, const char *argv[]);
  int gpu_id;
  int nbDims;
  cudnnConvolutionFwdAlgo_t convolution_algorithm;

  // {n, in_channels, input_height, input_width}
  int in_dimA[MAX_DIM];
  // {out_channels, in_channels, kernel_height, kernel_width}
  int ker_dim[MAX_DIM];
  int ker_pad[MAX_DIM];
  int ker_stride[MAX_DIM];
  int ker_dilation[MAX_DIM];

  // kinda fixed values
  cudnnTensorFormat_t input_format;
  cudnnTensorFormat_t output_format;
  // Note: if kernel_format is NHWC (i.e. not NCHW) then the
  // support is limited.
  cudnnTensorFormat_t kernel_format;
  cudnnConvolutionMode_t mode;
  cudnnDataType_t data_type = CUDNN_DATA_FLOAT;
  bool with_sigmoid;

  // Calculated and/or filled values:
  int tensor_dims;
  // {n, in_channels, input_height, input_width}
  int out_dimA[MAX_DIM]; // filled later

  const int nb_fixed_args = 7; // including scrip name!

  int getInputBytes() const;
  int getOutputBytes() const;
  int getKernelBytes() const;
  friend std::ostream &operator<<(std::ostream &os, const Args &args);

private:
  Args();
  int prod(const int *arr) const;
  void usage();
  void loadArgs(int *arr, int arg_batch);
  int argc;
  const char **argv;
  int cur_arg;
};

Args::Args(const int argc, const char *argv[])
    : in_dimA{1, 3, 578, 549}, ker_dim{3, 3, 3, 3}, ker_pad{1, 1},
      ker_stride{1, 1}, ker_dilation{1, 1}, input_format{CUDNN_TENSOR_NHWC},
      output_format{CUDNN_TENSOR_NHWC}, kernel_format{CUDNN_TENSOR_NCHW},
      mode{CUDNN_CROSS_CORRELATION}, data_type{CUDNN_DATA_FLOAT},
      with_sigmoid{false}, argc{argc}, argv{argv}, cur_arg{1} {
  if (argc < nb_fixed_args)
    usage();

  gpu_id = std::atoi(argv[1]);
  convolution_algorithm = cudnnConvolutionFwdAlgo_t(std::atoi(argv[2]));
  nbDims = std::atoi(argv[3]);
  int batch_size = std::atoi(argv[4]);
  int in_channels = std::atoi(argv[5]);
  int out_channels = std::atoi(argv[6]);

  tensor_dims = nbDims + 2;
  in_dimA[0] = batch_size;
  in_dimA[1] = in_channels;

  ker_dim[0] = out_channels;
  ker_dim[1] = in_channels;
  // ker_dim[2] = 7;
  // ker_dim[3] = 7;

  loadArgs(in_dimA + 2, 0);
  loadArgs(ker_dim + 2, 1);
  loadArgs(ker_pad, 2);
  loadArgs(ker_stride, 3);
  loadArgs(ker_dilation, 4);
}

int Args::getInputBytes() const { return prod(in_dimA) * sizeof(float); }

int Args::getOutputBytes() const { return prod(out_dimA) * sizeof(float); }

int Args::getKernelBytes() const { return prod(ker_dim) * sizeof(float); }

int Args::prod(const int *arr) const {
  int result = 1;
  for (int i = 0; i < tensor_dims; i++)
    result *= arr[i];
  return result;
}

void Args::usage() {
  std::cerr << "Usage: " << argv[0] << "  <gpu_id> <conv_algo> <nbDims> \\\n"
            << "  <batch_size> <in_ch> <out_ch> \\\n"
            << "  <inDim_1> .. <inDim_nbDims> \\\n"
            << "  <kerDim_1> .. <kerDim_nbDims> \\\n"
            << "  <pad_1> .. <pad_nbDims> \\\n"
            << "  <stride_1> .. <stride_nbDims> \\\n"
            << "  <dilation_1> .. <dilation_nbDims>" << std::endl;
  std::exit(EXIT_FAILURE);
}

void Args::loadArgs(int *arr, int arg_batch) {
  if (argc < nb_fixed_args + (arg_batch + 1) * nbDims) {
    return;
  }
  for (int i = 0; i < nbDims; i++) {
    const int idx = nb_fixed_args + arg_batch * nbDims + i;
    arr[i] = std::atoi(argv[idx]);
  }
}

std::ostream &operator<<(std::ostream &os, const Args &args) {
  os << "gpu_id: " << args.gpu_id << std::endl;
  os << "nbDims: " << args.nbDims << std::endl;
  os << "algo: " << int(args.convolution_algorithm) << std::endl;
  os << "in_dim: ";
  for (int i = 0; i < MAX_DIM; i++) {
    os << args.in_dimA[i] << ", ";
  }
  os << std::endl;
  os << "ker_dim: ";
  for (int i = 0; i < MAX_DIM; i++) {
    os << args.ker_dim[i] << ", ";
  }
  os << std::endl;
  os << "ker_pad: ";
  for (int i = 0; i < MAX_DIM; i++) {
    os << args.ker_pad[i] << ", ";
  }
  os << std::endl;
  os << "ker_stride: ";
  for (int i = 0; i < MAX_DIM; i++) {
    os << args.ker_stride[i] << ", ";
  }
  os << std::endl;
  os << "ker_dilation: ";
  for (int i = 0; i < MAX_DIM; i++) {
    os << args.ker_dilation[i] << ", ";
  }
  os << std::endl;
  os << "input_format: " << args.input_format << std::endl;
  os << "output_format: " << args.output_format << std::endl;
  os << "kernel_format: " << args.kernel_format << std::endl;
  os << "mode: " << args.mode << std::endl;
  os << "data_type: " << args.data_type << std::endl;
  os << "with_sigmoid: " << args.with_sigmoid << std::endl;
  os << "tensor_dims;: " << args.tensor_dims << std::endl;

  os << "out_dimA[]: ";
  for (int i = 0; i < MAX_DIM; i++) {
    os << args.out_dimA[i] << ", ";
  }
  os << std::endl;
  return os;
}

cudnnTensorDescriptor_t getInputDescriptor(const Args &args) {
  cudnnTensorDescriptor_t input_descriptor;
  checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
  checkCUDNN(cudnnSetTensorNdDescriptorEx(input_descriptor, args.input_format,
                                          args.data_type, args.tensor_dims,
                                          args.in_dimA));
  return input_descriptor;
}

cudnnFilterDescriptor_t getKernelDescriptor(const Args &args) {
  cudnnFilterDescriptor_t kernel_descriptor;
  checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
  checkCUDNN(cudnnSetFilterNdDescriptor(kernel_descriptor, args.data_type,
                                        args.kernel_format, args.tensor_dims,
                                        args.ker_dim));
  return kernel_descriptor;
}

cudnnConvolutionDescriptor_t getConvDescriptor(const Args &args) {
  cudnnConvolutionDescriptor_t convolution_descriptor;
  checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
  checkCUDNN(cudnnSetConvolutionNdDescriptor(
      convolution_descriptor, args.nbDims, args.ker_pad, args.ker_stride,
      args.ker_dilation, args.mode, args.data_type));
  return convolution_descriptor;
}

cudnnTensorDescriptor_t getOutputDescriptor(const Args &args) {
  cudnnTensorDescriptor_t output_descriptor;
  checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
  checkCUDNN(cudnnSetTensorNdDescriptorEx(output_descriptor, args.output_format,
                                          args.data_type, args.tensor_dims,
                                          args.out_dimA));
  return output_descriptor;
}

void fillInput(float *h_input, const size_t &input_elems) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);
  for (size_t i = 0; i < input_elems; i++) {
    h_input[i] = dis(gen);
  }
}

void fillKernel(float *h_kernel, const size_t &kernel_elems) {
  const float kernel_template[] = {1, 1, 1, 1, -8, 1, 1, 1, 1};
  int mod = (9 > kernel_elems ? kernel_elems : 9);
  for (int idx = 0; idx < kernel_elems; ++idx)
    h_kernel[idx] = kernel_template[idx % mod];
}

int main(int argc, const char *argv[]) {
  Args args(argc, argv);

  cudaSetDevice(args.gpu_id);

  cudnnHandle_t cudnn;
  cudnnCreate(&cudnn);

  cudnnTensorDescriptor_t input_descriptor = getInputDescriptor(args);
  cudnnFilterDescriptor_t kernel_descriptor = getKernelDescriptor(args);
  cudnnConvolutionDescriptor_t convolution_descriptor = getConvDescriptor(args);
  checkCUDNN(cudnnGetConvolutionNdForwardOutputDim(
      convolution_descriptor, input_descriptor, kernel_descriptor,
      args.tensor_dims, args.out_dimA));
  cudnnTensorDescriptor_t output_descriptor = getOutputDescriptor(args);

  // CUDNN_CONVOLUTION_FWD_ALGO_GEMM; checkCUDNN(
  // cudnnGetConvolutionForwardAlgorithm(cudnn, input_descriptor,
  // kernel_descriptor, convolution_descriptor, output_descriptor,
  // CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, /*memoryLimitInBytes=*/0,
  // &convolution_algorithm));

  size_t kernel_bytes = args.getKernelBytes();
  size_t kernel_elems = args.getKernelBytes() / sizeof(float);
  size_t input_bytes = args.getInputBytes();
  size_t input_elems = input_bytes / sizeof(float);
  size_t output_bytes = args.getOutputBytes();

  size_t workspace_bytes{0};
  checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(
      cudnn, input_descriptor, kernel_descriptor, convolution_descriptor,
      output_descriptor, args.convolution_algorithm, &workspace_bytes));

  float *h_kernel = new float[kernel_elems];
  float *h_input = new float[input_elems];

  void *d_workspace{nullptr};
  float *d_kernel{nullptr};
  float *d_input{nullptr};
  float *d_output{nullptr};
  cudaMalloc(&d_workspace, workspace_bytes);
  cudaMalloc(&d_kernel, kernel_bytes);
  cudaMalloc(&d_input, input_bytes);
  cudaMalloc(&d_output, output_bytes);

  fillKernel(h_kernel, kernel_elems);
  fillInput(h_input, input_elems);

  cudaMemcpy(d_kernel, h_kernel, kernel_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_input, h_input, input_bytes, cudaMemcpyHostToDevice);
  cudaMemset(d_output, 0, output_bytes);

  auto start = std::chrono::high_resolution_clock::now();
  const float alpha = 1.0f, beta = 0.0f;
  checkCUDNN(cudnnConvolutionForward(
      cudnn, &alpha, input_descriptor, d_input, kernel_descriptor, d_kernel,
      convolution_descriptor, args.convolution_algorithm, d_workspace,
      workspace_bytes, &beta, output_descriptor, d_output));
  cudaDeviceSynchronize();
  auto stop = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> seconds = (stop - start);
  std::cout << "'time': " << seconds.count() << std::endl;

  if (args.with_sigmoid) {
    cudnnActivationDescriptor_t activation_descriptor;
    checkCUDNN(cudnnCreateActivationDescriptor(&activation_descriptor));
    checkCUDNN(cudnnSetActivationDescriptor(
        activation_descriptor, CUDNN_ACTIVATION_SIGMOID, CUDNN_PROPAGATE_NAN,
        /*relu_coef=*/0));
    checkCUDNN(cudnnActivationForward(cudnn, activation_descriptor, &alpha,
                                      output_descriptor, d_output, &beta,
                                      output_descriptor, d_output));
    cudnnDestroyActivationDescriptor(activation_descriptor);
  }

  // float *h_output = new float[output_bytes];
  // cudaMemcpy(h_output, d_output, output_bytes, cudaMemcpyDeviceToHost);
  // cv::Mat output_image(args.out_dimA[2], args.out_dimA[3], CV_32FC3,
  // h_output); cv::threshold(output_image, output_image, 0, 0,
  // cv::THRESH_TOZERO); cv::normalize(output_image, output_image, 0.0, 255.0,
  // cv::NORM_MINMAX); output_image.convertTo(output_image, CV_8UC3);
  // cv::imwrite("cudnn-out.png", output_image);
  // delete[] h_output;

  delete[] h_kernel;
  delete[] h_input;
  cudaFree(d_workspace);
  cudaFree(d_kernel);
  cudaFree(d_input);
  cudaFree(d_output);

  cudnnDestroyTensorDescriptor(input_descriptor);
  cudnnDestroyTensorDescriptor(output_descriptor);
  cudnnDestroyFilterDescriptor(kernel_descriptor);
  cudnnDestroyConvolutionDescriptor(convolution_descriptor);

  cudnnDestroy(cudnn);
}
