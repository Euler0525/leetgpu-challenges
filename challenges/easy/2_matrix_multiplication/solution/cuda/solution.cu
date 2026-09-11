#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void matrix_multiplication_kernel(const float *A, const float *B,
                                             float *C, int M, int N, int K) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    float acc = 0.0f;
    for (int tile = 0; tile < (N + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        int Acol = tile * TILE_SIZE + threadIdx.x;
        if (row < M && Acol < N) {
            As[threadIdx.y][threadIdx.x] = A[row * N + Acol];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        int Brow = tile * TILE_SIZE + threadIdx.y;
        if (Brow < N && col < K) {
            Bs[threadIdx.y][threadIdx.x] = B[Brow * K + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int n = 0; n < TILE_SIZE; ++n) {
            acc += As[threadIdx.y][n] * Bs[n][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < M && col < K) {
        C[row * K + col] = acc;
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float *A, const float *B, float *C, int M, int N,
                      int K) {
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M,
                                                                     N, K);
    cudaDeviceSynchronize();
}