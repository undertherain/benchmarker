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
#include <cstdlib>
#include <cudnn.h>
#include <cudnn_cnn_infer.h>
#include <cudnn_ops_infer.h>
#include <iostream>
#include <opencv2/opencv.hpp>

#define checkCUDNN(expression)                                                 \
  {                                                                            \
    cudnnStatus_t status = (expression);                                       \
    if (status != CUDNN_STATUS_SUCCESS) {                                      \
      std::cerr << "Error on line " << __LINE__ << ": "                        \
                << cudnnGetErrorString(status) << std::endl;                   \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  }

cv::Mat load_image(const char *image_path) {
  cv::Mat image = cv::imread(image_path);
  image.convertTo(image, CV_32FC3);
  cv::normalize(image, image, 0, 1, cv::NORM_MINMAX);
  std::cerr << "Input Image: " << image.rows << " x " << image.cols << " x "
            << image.channels() << std::endl;
  return image;
}

void save_image(const char *output_filename, float *buffer, int height,
                int width) {
  cv::Mat output_image(height, width, CV_32FC3, buffer);
  cv::threshold(output_image, output_image,
                /*threshold=*/0,
                /*maxval=*/0, cv::THRESH_TOZERO);
  cv::normalize(output_image, output_image, 0.0, 255.0, cv::NORM_MINMAX);
  output_image.convertTo(output_image, CV_8UC3);
  cv::imwrite(output_filename, output_image);
  std::cerr << "Wrote output to " << output_filename << std::endl;
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

  const int nb_fixed_args = 7; // +!

  int getInputBytes();
  int getOutputBytes();
  int getKernelBytes();
  void setOutDims(const cudnnConvolutionDescriptor_t &convolution_descriptor,
                  const cudnnTensorDescriptor_t &input_descriptor,
                  const cudnnFilterDescriptor_t &kernel_descriptor);

private:
  Args();
  int prod(int *arr);
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

  loadArgs(in_dimA + 2, 0);
  loadArgs(ker_dim + 2, 1);
  loadArgs(ker_pad, 2);
  loadArgs(ker_stride, 3);
  loadArgs(ker_dilation, 4);
  std::cout << "ker[]: " << ker_dim[0] << ", " << ker_dim[1] << ", "
            << ker_dim[2] << ", " << ker_dim[3] << ", " << std::endl;
}

int Args::getInputBytes() { return prod(in_dimA) * sizeof(float); }

int Args::getOutputBytes() { return prod(out_dimA) * sizeof(float); }

int Args::getKernelBytes() { return prod(ker_dim) * sizeof(float); }

void Args::setOutDims(
    const cudnnConvolutionDescriptor_t &convolution_descriptor,
    const cudnnTensorDescriptor_t &input_descriptor,
    const cudnnFilterDescriptor_t &kernel_descriptor) {
  checkCUDNN(cudnnGetConvolutionNdForwardOutputDim(
      convolution_descriptor, input_descriptor, kernel_descriptor, tensor_dims,
      out_dimA));
}

int Args::prod(int *arr) {
  int result = 1;
  for (int i = 0; i < tensor_dims; i++)
    result *= arr[i];
  return result;
}

void Args::usage() {
  // 2d bs in_ch d0 d1 f0 f1 s0 s1 d0 d1 p0 p1
  // 3d bs in_ch d0 d1 d2
  std::cerr << "Usage: " << argv[0] << "  <gpu_id> <nbDims> <conv_algo> \\\n"
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
    arr[i] = std::atoi(argv[nb_fixed_args + arg_batch * nbDims + i]);
  }
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
                                          args.in_dimA));
  return output_descriptor;
}

void fillKernel(float *h_kernel, const size_t &kernel_elems) {
  const float kernel_template[] = {1, 1, 1, 1, -8, 1, 1, 1, 1};
  for (int idx = 0; idx < kernel_elems; ++idx)
    h_kernel[idx] = kernel_template[idx % 9];
}

void fillInput(float *h_input, const size_t &input_elems) {
  const float kernel_template[] = {1, 1, 1, 1, -8, 1, 1, 1, 1};
  for (int idx = 0; idx < input_elems; ++idx)
    h_input[idx] = kernel_template[idx % 9];
}

int main(int argc, const char *argv[]) {
  Args args(argc, argv);
  cv::Mat image = load_image("tensorflow.png");

  cudaSetDevice(args.gpu_id);

  cudnnHandle_t cudnn;
  cudnnCreate(&cudnn);

  cudnnTensorDescriptor_t input_descriptor = getInputDescriptor(args);
  cudnnFilterDescriptor_t kernel_descriptor = getKernelDescriptor(args);
  cudnnConvolutionDescriptor_t convolution_descriptor = getConvDescriptor(args);
  args.setOutDims(convolution_descriptor, input_descriptor, kernel_descriptor);
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
  float *old = h_input;
  h_input = image.ptr<float>(0);

  cudaMemcpy(d_kernel, h_kernel, kernel_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_input, h_input, input_bytes, cudaMemcpyHostToDevice);
  cudaMemset(d_output, 0, output_bytes);
  h_input = old;

  const float alpha = 1.0f, beta = 0.0f;
  checkCUDNN(cudnnConvolutionForward(
      cudnn, &alpha, input_descriptor, d_input, kernel_descriptor, d_kernel,
      convolution_descriptor, args.convolution_algorithm, d_workspace,
      workspace_bytes, &beta, output_descriptor, d_output));

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

  float *h_output = new float[output_bytes];
  cudaMemcpy(h_output, d_output, output_bytes, cudaMemcpyDeviceToHost);
  save_image("cudnn-out.png", h_output, args.out_dimA[2], args.out_dimA[3]);
  delete[] h_output;

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
