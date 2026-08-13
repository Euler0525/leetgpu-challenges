#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>

#define CUDA_CHECK(expr_to_check)                                              \
    do {                                                                       \
        cudaError_t result = expr_to_check;                                    \
        if (result != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA Runtime Error: %s:%i:%d = %s\n", __FILE__,   \
                    __LINE__, result, cudaGetErrorString(result));             \
        }                                                                      \
    } while (0)

__device__ __forceinline__ float warp_reduce_max(float val) {
#pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }

    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/**
 * Kernel1:
 * Q * K^T -> Scores
 * (M * d)(N * d) -> (M * N)
 *
 */
__global__ void matmul_qk(const float *Q, const float *K, float *Scores, int M,
                          int N, int d) {
    int x = blockDim.x * blockIdx.x + threadIdx.x; // [0, N)
    int y = blockDim.y * blockIdx.y + threadIdx.y; // [0, M)
    if (x >= N || y >= M) {
        return;
    }
    float dot = 0.0f;
    for (int i = 0; i < d; ++i) {
        dot += Q[y * d + i] * K[x * d + i];
    }
    Scores[y * N + x] = dot / sqrtf((float)d);
}

/**
 * Kernel2: Row-wise Softmax on Scores[M*N]
 * Each block processes one row (M blocks total)
 * Uses shared memory for block-level reduction
 */
__global__ void softmax_kernel(float *Scores, int M, int N) {
    int row = blockIdx.x;
    if (row >= M) {
        return;
    }

    // Find row max
    int tid = threadIdx.x;
    int lane = tid % warpSize;
    int warp_id = tid / warpSize;
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    /**
     * Thread 0: col = 0, 256, 512, 768
     * Thread 1: col = 1, 257, 513, 769
     * ...
     * Thread 255: col = 255, 511, 767, 1023
     */
    float max_val = -INFINITY;
    for (int col = tid; col < N; col += blockDim.x) {
        max_val = fmaxf(max_val, Scores[row * N + col]);
    }

    // // Block-level reduction for max
    // extern __shared__ float shared[];
    // shared[tid] = max_val;
    // __syncthreads();

    // for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    //     if (tid < stride) {
    //         shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
    //     }
    //     __syncthreads();
    // }
    // max_val = shared[0];
    max_val = warp_reduce_max(max_val);
    extern __shared__ float shared[];
    if (lane == 0) {
        shared[warp_id] = max_val; // lan0 contains max value
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (tid < num_warps) ? shared[tid] : -INFINITY;
        val = warp_reduce_max(val);
        if (tid == 0) {
            shared[0] = val;
        }
    }
    __syncthreads();
    max_val = shared[0];

    // Compute exp(x - max) and sum
    float sum_val = 0.0f;
    for (int col = tid; col < N; col += blockDim.x) {
        float exp_val = expf(Scores[row * N + col] - max_val);
        Scores[row * N + col] = exp_val;
        sum_val += exp_val;
    }

    // // Block-level reduction for sum
    // shared[tid] = sum_val;
    // __syncthreads();
    // for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    //     if (tid < stride) {
    //         shared[tid] += shared[tid + stride];
    //     }
    //     __syncthreads();
    // }
    // sum_val = shared[0];
    // __syncthreads();
    sum_val = warp_reduce_sum(sum_val);

    // Warp reduction
    if (lane == 0) {
        shared[warp_id] = sum_val;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (tid < num_warps) ? shared[tid] : 0.0f;
        val = warp_reduce_sum(val);
        if (tid == 0) {
            shared[0] = val;
        }
    }
    __syncthreads();
    float total_sum = shared[0];

    // Normalize
    float inv_sum = 1.0f / total_sum;
    for (int col = tid; col < N; col += blockDim.x) {
        Scores[row * N + col] *= inv_sum;
    }
}

/**
 * Kernel3: Scores * V -> output
 * Scores: [M, N], V: [N, d] -> output: [M, d]
 */
__global__ void matmul_sv(const float *Scores, const float *V, float *output,
                          int M, int N, int d) {
    int col = blockDim.x * blockIdx.x + threadIdx.x; // [0, d)
    int row = blockDim.y * blockIdx.y + threadIdx.y; // [0, M)
    if (col >= d || row >= M) {
        return;
    }

    float sum = 0.0f;
    for (int k = 0; k < N; ++k) {
        sum += Scores[row * N + k] * V[k * d + col];
    }
    output[row * d + col] = sum;
}

// Q, K, V, output are device pointers
extern "C" void solve(const float *Q, const float *K, const float *V,
                      float *output, int M, int N, int d) {
    // Allocate temporary Scores buffer [M, N]
    float *Scores;
    CUDA_CHECK(cudaMalloc(&Scores, M * N * sizeof(float)));

    // === Kernel 1: Q * K^T ===
    dim3 block_qk(16, 16);
    dim3 grid_qk((N + 15) / 16, (M + 15) / 16);
    matmul_qk<<<grid_qk, block_qk>>>(Q, K, Scores, M, N, d);
    CUDA_CHECK(cudaGetLastError());

    // === Kernel 2: Softmax ===
    const int softmax_block_size = 256; // Tune based on N
    size_t shared_mem_size = softmax_block_size * sizeof(float);
    softmax_kernel<<<M, softmax_block_size, shared_mem_size>>>(Scores, M, N);
    CUDA_CHECK(cudaGetLastError());

    // === Kernel 3: Scores * V ===
    dim3 block_sv(16, 16);
    dim3 grid_sv((d + 15) / 16, (M + 15) / 16);
    matmul_sv<<<grid_sv, block_sv>>>(Scores, V, output, M, N, d);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(Scores));
}