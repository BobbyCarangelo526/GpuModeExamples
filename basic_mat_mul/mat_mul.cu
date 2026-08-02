#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <iostream>
#include <cstdio>

__global__
void MatMulKernel(int64_t *A, int64_t *B, int64_t *C, int m, int n, int k) {
    // printf("Hello from block index x: %d y: %d.\nthread index x: %d y: %d.\nblock_dim x : %d, block_dim y: %d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, blockDim.x, blockDim.y);
    int output_index_x = blockDim.x * blockIdx.x + threadIdx.x;
    int output_index_y = blockDim.y * blockIdx.y + threadIdx.y;

    if (output_index_x >= m || output_index_y >= n) {
        return;
    }

    int row_offset_a = output_index_x * k;
    int col_offset_b = output_index_y;

    int output_offset = (n * output_index_x) + (output_index_y);
    int64_t &output_ref = *(C + output_offset);
    output_ref = 0;


    for (int i = 0; i < k; i++) {
        int64_t &value_A_ref = *(A + row_offset_a + i);
        int64_t &value_B_ref = *(B + col_offset_b + (i * n));
        output_ref += value_A_ref * value_B_ref;
    }
}

inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b; }

torch::Tensor mat_mul(torch::Tensor tensor_1, torch::Tensor tensor_2) {
    // CHECKS on tensors
    TORCH_CHECK(tensor_1.dim() == 2);
    TORCH_CHECK(tensor_2.dim() == 2);
    TORCH_CHECK(tensor_1.size(1) == tensor_2.size(0));
    
    for (int i = 0; i < 2; i++) {
        TORCH_CHECK(tensor_1.size(i) > 0);
        TORCH_CHECK(tensor_2.size(i) > 0);
    }

    // make contiguous
    torch::Tensor A_contiguous = tensor_1.contiguous();  
    torch::Tensor B_contiguous = tensor_2.contiguous();  
    int64_t *A_h = A_contiguous.data_ptr<int64_t>();
    int64_t *B_h = B_contiguous.data_ptr<int64_t>();

    // get M, K, N
    int M = tensor_1.size(0);
    int K = tensor_1.size(1);
    int N = tensor_2.size(1);

    // allocate device memory
    int64_t *A_d, *B_d, *C_d;

    size_t size_A_bytes = M * K * sizeof(int64_t);
    size_t size_B_bytes = K * N * sizeof(int64_t);
    size_t size_C_bytes = M * N * sizeof(int64_t);

    std::cout << "M, N, K: " << M << "," << N << "," << K << std::endl;
    std::cout << "Size bytes A: " << size_A_bytes << std::endl;
    std::cout << "Size bytes B: " << size_B_bytes << std::endl;
    std::cout << "Size bytes C: " << size_C_bytes << std::endl;

    cudaMalloc((void**) &A_d, size_A_bytes);
    cudaMalloc((void**) &B_d, size_B_bytes);
    cudaMalloc((void**) &C_d, size_C_bytes);

    // copy input matrices to device
    cudaMemcpy(A_d, A_h, size_A_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size_B_bytes, cudaMemcpyHostToDevice);

    // calculate gridDim and blockDim
    constexpr int kBlockDimX = 2;
    constexpr int kBlockDimY = 2;
    const int kGridDimX = cdiv(M, kBlockDimX);
    const int kGridDimY = cdiv(N, kBlockDimY);

    std::cout << "Block Dim: X: " << kBlockDimX << " Y: " << kBlockDimY << std::endl;
    std::cout << "Grid Dim: X: " << kGridDimX << " Y: " << kGridDimY << std::endl;

    dim3 blockDim(kBlockDimX, kBlockDimY, 1);
    dim3 gridDim(kGridDimX, kGridDimY, 1);

    // invoke kernel
    MatMulKernel<<<gridDim, blockDim>>>(A_d, B_d, C_d, M, N, K);

    // copy output matrix to host
    torch::Tensor result = torch::empty({M, N}, torch::kInt64);
    cudaMemcpy(result.data_ptr<int64_t>(), C_d, size_C_bytes, cudaMemcpyDeviceToHost);

    // free device memory
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    // return output tensor
    return result;
}