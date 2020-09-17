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
  int nbDims;
  // {n, in_channels, input_height, input_width}
  int in_dimA[MAX_DIM];
  int out_dimA[MAX_DIM];
  int getInputBytes();
  int getOutputBytes();

  // {out_channels, in_channels, kernel_height, kernel_width}
  int ker_dim[MAX_DIM];
  int ker_len;
  int ker_pad[MAX_DIM];
  int ker_stride[MAX_DIM];
  int ker_dilation[MAX_DIM];

  int kernel_height = 3;
  int kernel_width = 3;

private:
  Args();
  int prod(int *arr);
};

Args::Args(const int argc, const char *argv[])
    : nbDims{4}, in_dimA{1, 3, 578, 549}, ker_dim{3, 3, 3, 3},
      ker_len{nbDims - 2}, ker_pad{1, 1}, ker_stride{1, 1}, ker_dilation{1, 1} {
  if (argc < 2) {
    // 2d bs in_ch d0 d1 f0 f1 s0 s1 d0 d1 p0 p1
    // 3d bs in_ch d0 d1 d2
    std::cerr << "usage: " << argv[0]
              << " "
                 "gpu"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

int Args::getInputBytes() { return prod(in_dimA) * sizeof(float); }

int Args::getOutputBytes() { return prod(out_dimA) * sizeof(float); }

int Args::prod(int *arr) {
  int result = 1;
  for (int i = 0; i < nbDims; i++)
    result *= arr[i];
  return result;
}

int main(int argc, const char *argv[]) {
  Args args(argc, argv);
  int algo = (argc > 2) ? std::atoi(argv[2]) : CUDNN_CONVOLUTION_FWD_ALGO_GEMM;
  std::cerr << "Algorighm: " << algo << std::endl;

  int gpu_id = (argc > 3) ? std::atoi(argv[3]) : 0;
  std::cerr << "GPU: " << gpu_id << std::endl;

  bool with_sigmoid = (argc > 4) ? std::atoi(argv[4]) : 0;
  std::cerr << "With sigmoid: " << std::boolalpha << with_sigmoid << std::endl;

  cv::Mat image = load_image("tensorflow.png");

  cudnnDataType_t data_type = CUDNN_DATA_FLOAT;

  cudnnTensorFormat_t input_format = CUDNN_TENSOR_NHWC;
  int in_channels = 3;
  int input_height = image.rows;
  int input_width = image.cols;

  cudnnTensorFormat_t output_format = CUDNN_TENSOR_NHWC;
  int out_channels = 3;

  // Note: if kernel_format is NHWC (i.e. not NCHW) then the
  // support is limited.
  cudnnTensorFormat_t kernel_format = CUDNN_TENSOR_NCHW;
  cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION;

  cudaSetDevice(gpu_id);

  cudnnHandle_t cudnn;
  cudnnCreate(&cudnn);

  cudnnTensorDescriptor_t input_descriptor;
  checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
  checkCUDNN(cudnnSetTensorNdDescriptorEx(
      input_descriptor, input_format, data_type, args.nbDims, args.in_dimA));

  cudnnFilterDescriptor_t kernel_descriptor;
  checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
  checkCUDNN(cudnnSetFilterNdDescriptor(
      kernel_descriptor, data_type, kernel_format, args.nbDims, args.ker_dim));

  cudnnConvolutionDescriptor_t convolution_descriptor;
  checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));

  checkCUDNN(cudnnSetConvolutionNdDescriptor(
      convolution_descriptor, args.ker_len, args.ker_pad, args.ker_stride,
      args.ker_dilation, mode, data_type));

  checkCUDNN(cudnnGetConvolutionNdForwardOutputDim(
      convolution_descriptor, input_descriptor, kernel_descriptor, args.nbDims,
      args.out_dimA));

  cudnnTensorDescriptor_t output_descriptor;
  checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
  checkCUDNN(cudnnSetTensorNdDescriptorEx(
      output_descriptor, output_format, data_type, args.nbDims, args.in_dimA));

  cudnnConvolutionFwdAlgo_t convolution_algorithm =
      cudnnConvolutionFwdAlgo_t(algo);
  // CUDNN_CONVOLUTION_FWD_ALGO_GEMM;
  // checkCUDNN(
  //     cudnnGetConvolutionForwardAlgorithm(cudnn,
  //                                         input_descriptor,
  //                                         kernel_descriptor,
  //                                         convolution_descriptor,
  //                                         output_descriptor,
  //                                         CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
  //                                         /*memoryLimitInBytes=*/0,
  //                                         &convolution_algorithm));

  size_t workspace_bytes{0};
  checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(
      cudnn, input_descriptor, kernel_descriptor, convolution_descriptor,
      output_descriptor, convolution_algorithm, &workspace_bytes));
  std::cerr << "Workspace size: " << (workspace_bytes / 1048576.0) << "MB"
            << std::endl;
  // assert(workspace_bytes > 0);

  void *d_workspace{nullptr};
  cudaMalloc(&d_workspace, workspace_bytes);

  int input_bytes = args.getInputBytes();
  int output_bytes = args.getOutputBytes();

  float *d_input{nullptr};
  cudaMalloc(&d_input, input_bytes);
  cudaMemcpy(d_input, image.ptr<float>(0), input_bytes, cudaMemcpyHostToDevice);

  float *d_output{nullptr};
  cudaMalloc(&d_output, output_bytes);
  cudaMemset(d_output, 0, output_bytes);

  // clang-format off
  const float kernel_template[3][3] = {
    {1, 1, 1},
    {1, -8, 1},
    {1, 1, 1}
  };
  // clang-format on

  float h_kernel[3][3][3][3];
  for (int kernel = 0; kernel < 3; ++kernel) {
    for (int channel = 0; channel < 3; ++channel) {
      for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
          h_kernel[kernel][channel][row][column] = kernel_template[row][column];
        }
      }
    }
  }

  float *d_kernel{nullptr};
  cudaMalloc(&d_kernel, sizeof(h_kernel));
  cudaMemcpy(d_kernel, h_kernel, sizeof(h_kernel), cudaMemcpyHostToDevice);

  const float alpha = 1.0f, beta = 0.0f;

  checkCUDNN(cudnnConvolutionForward(
      cudnn, &alpha, input_descriptor, d_input, kernel_descriptor, d_kernel,
      convolution_descriptor, convolution_algorithm, d_workspace,
      workspace_bytes, &beta, output_descriptor, d_output));

  if (with_sigmoid) {
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

  save_image("cudnn-out.png", h_output, input_height, input_width);

  delete[] h_output;
  cudaFree(d_kernel);
  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_workspace);

  cudnnDestroyTensorDescriptor(input_descriptor);
  cudnnDestroyTensorDescriptor(output_descriptor);
  cudnnDestroyFilterDescriptor(kernel_descriptor);
  cudnnDestroyConvolutionDescriptor(convolution_descriptor);

  cudnnDestroy(cudnn);
}
