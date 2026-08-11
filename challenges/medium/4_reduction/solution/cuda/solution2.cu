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
    __shared__ float shared[BLOCK_SIZE];
    float sum = 0.0f;
    if (idx < N) {
        sum += input[idx];
    }
    if (idx + BLOCK_SIZE < N) {
        sum += input[idx + BLOCK_SIZE];
    }
    shared[threadIdx.x] = sum;
    __syncthreads();

    for (int offset = BLOCK_SIZE >> 1; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            shared[threadIdx.x] += shared[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(output, shared[0]);
    }
}

extern "C" void solve(const float *input, float *output, int N) {
    cudaMemset(output, 0, sizeof(float));
    if (N <= 0) {
        return;
    }

    int elementsPerBlock = BLOCK_SIZE * 2;
    int BlocksPerGrid = (N + elementsPerBlock - 1) / elementsPerBlock;

    reduceKernel<<<BlocksPerGrid, BLOCK_SIZE>>>(input, output, N);
}
