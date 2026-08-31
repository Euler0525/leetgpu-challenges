#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

#define BLOCK_SIZE 256
#define WARPS_NUM (BLOCK_SIZE / 32)

__device__ __forceinline__ float warp_reduce_sum(float value) {
    value += __shfl_down_sync(0xffffffffu, value, 16);
    value += __shfl_down_sync(0xffffffffu, value, 8);
    value += __shfl_down_sync(0xffffffffu, value, 4);
    value += __shfl_down_sync(0xffffffffu, value, 2);
    value += __shfl_down_sync(0xffffffffu, value, 1);

    return value;
}

__device__ __forceinline__ float warp_reduce_max(float value) {
    value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, 16));
    value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, 8));
    value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, 4));
    value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, 2));
    value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, 1));

    return value;
}

__device__ __forceinline__ float softmax_max_kernel(float local_max_value,
                                                    float *shared) {
    int tid = threadIdx.x;
    local_max_value = warp_reduce_max(local_max_value);

    int lane_id = tid & 31;
    int warp_id = tid >> 5;

    if (lane_id == 0) {
        shared[warp_id] = local_max_value;
    }
    __syncthreads();

    if (warp_id == 0) {
        local_max_value = lane_id < WARPS_NUM ? shared[lane_id] : -FLT_MAX;
        local_max_value = warp_reduce_max(local_max_value);
        if (lane_id == 0) {
            shared[0] = local_max_value;
        }
    }
    __syncthreads();

    return shared[0];
}

__device__ __forceinline__ float softmax_sum_kernel(float local_sum_value,
                                                    float *shared) {
    int tid = threadIdx.x;
    local_sum_value = warp_reduce_sum(local_sum_value);

    int lane_id = tid & 31;
    int warp_id = tid >> 5;

    if (lane_id == 0) {
        shared[warp_id] = local_sum_value;
    }
    __syncthreads();

    if (warp_id == 0) {
        local_sum_value = lane_id < WARPS_NUM ? shared[lane_id] : 0.0f;
        local_sum_value = warp_reduce_sum(local_sum_value);
        if (lane_id == 0) {
            shared[0] = local_sum_value;
        }
    }
    __syncthreads();

    return shared[0];
}

template <int ELEMENTS_PER_THREAD>
__global__ void softmax_kernel(const float *input, float *output, int N) {
    int tid = threadIdx.x;
    __shared__ float shared[WARPS_NUM];
    float values[ELEMENTS_PER_THREAD];

    float local_max_value = -FLT_MAX;
#pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i += 1) {
        int idx = tid + i * BLOCK_SIZE;
        values[i] = idx < N ? input[idx] : -FLT_MAX;
        local_max_value = fmaxf(local_max_value, values[i]);
    }
    float max_value = softmax_max_kernel(local_max_value, shared);

    float local_sum_value = 0.0f;
#pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i += 1) {
        int idx = tid + i * BLOCK_SIZE;
        if (idx < N) {
            values[i] = expf(values[i] - max_value);
            local_sum_value += values[i];
        }
    }
    float inv_sum_value = 1.0f / softmax_sum_kernel(local_sum_value, shared);

#pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i += 1) {
        int idx = tid + i * BLOCK_SIZE;
        if (idx < N) {
            output[idx] = values[i] * inv_sum_value;
        }
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float *input, float *output, int N) {
    if (input == nullptr || output == nullptr || N <= 0) {
        return;
    }

    int elements_per_thread = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    softmax_kernel<16><<<1, BLOCK_SIZE>>>(input, output, N);
    cudaDeviceSynchronize();
}