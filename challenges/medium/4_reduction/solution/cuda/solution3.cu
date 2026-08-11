#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__device__ __forceinline__ float warpReduceSum(float value) {
    value += __shfl_down_sync(0xffffffff, value, 16);
    value += __shfl_down_sync(0xffffffff, value, 8);
    value += __shfl_down_sync(0xffffffff, value, 4);
    value += __shfl_down_sync(0xffffffff, value, 2);
    value += __shfl_down_sync(0xffffffff, value, 1);

    return value;
}

__global__ void reduceKernel(const float *input, float *output, int N) {
    int idx = blockDim.x * blockIdx.x * 2 + threadIdx.x;
    int stride = blockDim.x * gridDim.x * 2;
    __shared__ float shared[BLOCK_SIZE];

    // Grid-Stride Loop
    float sum = 0.0f;
    for (int i = idx; i < N; i += stride) {
        sum += input[i];
        if (i + BLOCK_SIZE < N) {
            sum += input[i + BLOCK_SIZE];
        }
    }

    shared[threadIdx.x] = sum;
    __syncthreads();

    for (int offset = BLOCK_SIZE >> 1; offset >= 32; offset >>= 1) {
        if (threadIdx.x < offset) {
            shared[threadIdx.x] += shared[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x < 32) {
        float value = shared[threadIdx.x];
        value = warpReduceSum(value);
        if (threadIdx.x == 0) {
            atomicAdd(output, value);
        }
    }
}

extern "C" void solve(const float *input, float *output, int N) {
    cudaMemset(output, 0, sizeof(float));
    if (N <= 0) {
        return;
    }

    int elementsPerBlock = BLOCK_SIZE * 2;
    int BlocksPerGrid = (N + elementsPerBlock - 1) / elementsPerBlock;
    BlocksPerGrid = BlocksPerGrid > 1024 ? 1024 : BlocksPerGrid;

    reduceKernel<<<BlocksPerGrid, BLOCK_SIZE>>>(input, output, N);
}
