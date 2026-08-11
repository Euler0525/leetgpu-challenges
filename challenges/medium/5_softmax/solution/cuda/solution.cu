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

__device__ __forceinline__ float block_reduce_max(float local_max_value) {
    // __shared__ float shared[BLOCK_SIZE];
    // shared[threadIdx.x] = local_max_value;
    // __syncthreads();

    // for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
    //     if (stride > threadIdx.x) {
    //         shared[threadIdx.x] =
    //             fmax(shared[threadIdx.x], shared[threadIdx.x + stride]);
    //     }
    //     __syncthreads();
    // }

    __shared__ float shared[WARPS_NUM];
    int lane_id = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    local_max_value = warp_reduce_max(local_max_value);
    if (lane_id == 0) {
        shared[warp_id] = local_max_value;
    }
    __syncthreads(); // 这里存在跨 warp 的 "生产者-消费者" 关系, 每个 warp 的
                     // lane0 是生产者, 第一个 warp 是消费者; 不同 warp
                     // 的执行进度没有保证, 这里必须同步.

    local_max_value = threadIdx.x < WARPS_NUM ? shared[lane_id] : -FLT_MAX;
    if (warp_id == 0) {
        local_max_value = warp_reduce_max(local_max_value);
    }

    if (threadIdx.x == 0) {
        shared[0] = local_max_value;
    }
    __syncthreads(); // 只在线程 0 写入最终结果, Block
                     // 中所有线程需要读取共享内存.

    return shared[0];
}

__device__ float block_reduce_sum(float local_sum_value) {
    // __shared__ float shared[BLOCK_SIZE];
    // shared[threadIdx.x] = local_sum_value;
    // __syncthreads();

    // for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
    //     if (stride > threadIdx.x) {
    //         shared[threadIdx.x] += shared[threadIdx.x + stride];
    //     }
    //     __syncthreads();
    // }
    __shared__ float shared[WARPS_NUM];
    local_sum_value = warp_reduce_sum(local_sum_value);
    int lane_id = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    if (lane_id == 0) {
        shared[warp_id] = local_sum_value;
    }
    __syncthreads();

    local_sum_value = threadIdx.x < WARPS_NUM ? shared[lane_id] : 0.0f;
    if (warp_id == 0) {
        local_sum_value = warp_reduce_sum(local_sum_value);
    }

    if (threadIdx.x == 0) {
        shared[0] = local_sum_value;
    }
    __syncthreads();

    return shared[0];
}

__global__ void softmax_kernel(const float *input, float *output, int N) {
    float local_max_value = -FLT_MAX;
    for (int i = threadIdx.x; i < N; i += BLOCK_SIZE) {
        local_max_value = fmaxf(local_max_value, input[i]);
    }

    float max_value = block_reduce_max(local_max_value);

    float local_sum_value = 0.0f;
    for (int i = threadIdx.x; i < N; i += BLOCK_SIZE) {
        float value = expf(input[i] - max_value);
        output[i] = value;
        local_sum_value += value;
    }

    float inv_sum = 1.0f / block_reduce_sum(local_sum_value);

    for (int i = threadIdx.x; i < N; i += BLOCK_SIZE) {
        output[i] *= inv_sum;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float *input, float *output, int N) {
    if (input == nullptr || output == nullptr || N <= 0) {
        return;
    }

    softmax_kernel<<<1, BLOCK_SIZE>>>(input, output, N);
    cudaDeviceSynchronize();
}