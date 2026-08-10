#include <cuda_runtime.h>

__global__ void silu_kernel(const float *input, float *output, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) {
        return;
    }

    output[idx] = input[idx] / (1 + exp(-1 * input[idx]));
}

// input, output are device pointers
extern "C" void solve(const float *input, float *output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    silu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
