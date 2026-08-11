#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define ELEMENTS_PER_THREAD 8

__device__ __forceinline__ float warpReduceSum(float val) {
    val += __shfl_down_sync(0xFFFFFFFF, val, 16);
    val += __shfl_down_sync(0xFFFFFFFF, val, 8);
    val += __shfl_down_sync(0xFFFFFFFF, val, 4);
    val += __shfl_down_sync(0xFFFFFFFF, val, 2);
    val += __shfl_down_sync(0xFFFFFFFF, val, 1);

    return val; // lane 0
}

__global__ __launch_bounds__(BLOCK_SIZE) void reduceKernel(
    const float *__restrict__ input, float *__restrict__ output, int N) {
    __shared__ float warpSums[BLOCK_SIZE / 32]; // 8 float

    float sum = 0.0f;
    int base = blockIdx.x * (BLOCK_SIZE * ELEMENTS_PER_THREAD) + threadIdx.x;
#pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
        int idx = base + i * BLOCK_SIZE;
        if (idx < N) {
            sum += input[idx];
        }
    }
    sum = warpReduceSum(sum);
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    if (lane == 0) {
        warpSums[wid] = sum;
    }
    __syncthreads();

    if (wid == 0) {
        sum = (threadIdx.x < (BLOCK_SIZE / 32)) ? warpSums[lane] : 0.0f;
        sum = warpReduceSum(sum);

        if (threadIdx.x == 0) {
            output[blockIdx.x] = sum;
        }
    }
}

__global__ __launch_bounds__(BLOCK_SIZE) void reduceFinalKernel(
    const float *__restrict__ input, float *__restrict__ output, int N) {
    __shared__ float warpSums[BLOCK_SIZE / 32];

    float sum = 0.0f;
    for (int i = blockIdx.x * BLOCK_SIZE + threadIdx.x; i < N;
         i += blockDim.x * gridDim.x) {
        sum += input[i];
    }

    sum = warpReduceSum(sum);

    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    if (lane == 0) {
        warpSums[wid] = sum;
    }
    __syncthreads();

    if (wid == 0) {
        sum = (threadIdx.x < (BLOCK_SIZE / 32)) ? warpSums[lane] : 0.0f;
        sum = warpReduceSum(sum);

        if (threadIdx.x == 0) {
            atomicAdd(output, sum);
        }
    }
}

extern "C" void solve(const float *input, float *output, int N) {
    if (N <= 0) {
        cudaMemset(output, 0, sizeof(float));
        return;
    }

    int elemsPerBlock = BLOCK_SIZE * ELEMENTS_PER_THREAD;    // 256 * 8 = 2048
    int numBlocks = (N + elemsPerBlock - 1) / elemsPerBlock; // Round up

    if (numBlocks == 1) {
        reduceKernel<<<1, BLOCK_SIZE>>>(input, output, N);
    } else {
        float *partial = nullptr;
        cudaMalloc(&partial, numBlocks * sizeof(float));

        reduceKernel<<<numBlocks, BLOCK_SIZE>>>(input, partial, N);
        int finalBlocks = (numBlocks + elemsPerBlock - 1) / elemsPerBlock;
        if (finalBlocks <= 1) {
            reduceKernel<<<1, BLOCK_SIZE>>>(partial, output, numBlocks);
        } else {
            cudaMemset(output, 0, sizeof(float));
            reduceFinalKernel<<<finalBlocks, BLOCK_SIZE>>>(partial, output,
                                                           numBlocks);
        }

        cudaFree(partial);
    }
}