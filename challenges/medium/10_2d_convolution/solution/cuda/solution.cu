#include <cuda_runtime.h>

#define BLOCK_SIZE 16
#define MAX_KERNEL_SIZE 4096

__constant__ float c_kernel[MAX_KERNEL_SIZE];

__global__ void conv2d_kernel(const float *input, const float *kernel,
                              float *output, int input_rows, int input_cols,
                              int kernel_rows, int kernel_cols, int output_rows,
                              int output_cols) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= output_rows || col >= output_cols) {
        return;
    }

    float acc = 0.0f;
    for (int kr = 0; kr < kernel_rows; ++kr) {
        for (int kc = 0; kc < kernel_cols; ++kc) {
            float x = input[(row + kr) * input_cols + (col + kc)];
            float k = c_kernel[kr * kernel_cols + kc];
            acc += x * k;
        }
    }

    output[row * output_cols + col] = acc;
}

// input, kernel, output are device pointers
extern "C" void solve(const float *input, const float *kernel, float *output,
                      int input_rows, int input_cols, int kernel_rows,
                      int kernel_cols) {
    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;

    int kernel_size = kernel_rows * kernel_cols;
    cudaMemcpyToSymbol(c_kernel, kernel, kernel_size * sizeof(float), 0,
                       cudaMemcpyDeviceToDevice);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((output_cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
                       (output_rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

    conv2d_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        input, kernel, output, input_rows, input_cols, kernel_rows, kernel_cols,
        output_rows, output_cols);
}
