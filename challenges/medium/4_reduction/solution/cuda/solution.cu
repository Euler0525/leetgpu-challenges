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
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int tid = threadIdx.x;
    __shared__ float shared[BLOCK_SIZE / 32];

    double sum = 0.0f;
    const float4 *input4 = reinterpret_cast<const float4 *>(input);
    for (int i = idx; i < N / 4; i += stride) {
        float4 value = input4[i];
        sum += value.x + value.y + value.z + value.w;
    }

    int tail = N / 4 * 4;
    for (int i = idx + tail; i < N; i += stride) {
        sum += input[i];
    }

    sum = warpReduceSum(sum);

    int lane_id = tid % 32;
    int warp_id = tid >> 5;

    if (lane_id == 0) {
        shared[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        double warp_sum = lane_id < BLOCK_SIZE / 32 ? shared[lane_id] : 0.0f;
        double block_sum = warpReduceSum(warp_sum);
        if (lane_id == 0) {
            atomicAdd(output, block_sum);
        }
    }
}

extern "C" void solve(const float *input, float *output, int N) {
    cudaMemset(output, 0, sizeof(float));
    if (N <= 0) {
        return;
    }

    int ThreadsPerBlock = BLOCK_SIZE;
    int elementsPerBlock = ThreadsPerBlock;
    int BlocksPerGrid = (N + elementsPerBlock - 1) / elementsPerBlock;
    BlocksPerGrid = BlocksPerGrid > 1024 ? 1024 : BlocksPerGrid;

    reduceKernel<<<BlocksPerGrid, ThreadsPerBlock>>>(input, output, N);
}