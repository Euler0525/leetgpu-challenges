#include <cuda_runtime.h>

__global__ void geglu_kernel(const float *input, float *output, int halfN) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < halfN) {
        output[idx] = input[idx] * 0.5f * input[idx + halfN] *
                      (1.0f + erff(input[idx + halfN] * 0.7071067811865475f));
    }
}

// input, output are device pointers
extern "C" void solve(const float *input, float *output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

    geglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    cudaDeviceSynchronize();
}
