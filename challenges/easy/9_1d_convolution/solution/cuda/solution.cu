#include <cuda_runtime.h>

__global__ void convolution_1d_kernel(const float *input, const float *kernel,
                                      float *output, int input_size,
                                      int kernel_size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int output_size = input_size - kernel_size + 1;
    extern __shared__ float shared_kernel[];
    for (int i = threadIdx.x; i < kernel_size; i += blockDim.x) {
        shared_kernel[i] = kernel[i];
    }
    __syncthreads();

    if (idx < output_size) {
        float acc = 0.0f;
#pragma unroll
        for (int i = 0; i < kernel_size; ++i) {
            acc += __ldg(&input[idx + i]) * shared_kernel[i];
        }
        output[idx] = acc;
    }
}

// input, kernel, output are device pointers (i.e. pointers to memory on the
// GPU)
extern "C" void solve(const float *input, const float *kernel, float *output,
                      int input_size, int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;
    size_t sharedMemSize = kernel_size * sizeof(float);

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
        input, kernel, output, input_size, kernel_size);
    cudaDeviceSynchronize();
}
