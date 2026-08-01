#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

__global__
void blur_kernel(unsigned char* output, const unsigned char* input,
                 int width, int height, int radius) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= width || row >= height) return;

    int totalR = 0, totalG = 0, totalB = 0, count = 0;
    for (int j = row - radius; j <= row + radius; ++j) {
        for (int i = col - radius; i <= col + radius; ++i) {
            if (i < 0 || i >= width || j < 0 || j >= height) continue;
            int off = 3 * (j * width + i);
            totalR += input[off + 0];
            totalG += input[off + 1];
            totalB += input[off + 2];
            ++count;
        }
    }
    int out = 3 * (row * width + col);
    output[out + 0] = (unsigned char)(totalR / count);
    output[out + 1] = (unsigned char)(totalG / count);
    output[out + 2] = (unsigned char)(totalB / count);
}

inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b; }

torch::Tensor blur_image(torch::Tensor image, int radius) {
    TORCH_CHECK(image.is_cuda(),                        "image must be a CUDA tensor");
    TORCH_CHECK(image.dtype() == torch::kByte,          "image must be uint8");
    TORCH_CHECK(image.dim() == 3 && image.size(2) == 3, "image must be HxWx3");
    image = image.contiguous();  // guarantee interleaved HWC memory

    const auto height = image.size(0);
    const auto width  = image.size(1);
    auto result = torch::empty({height, width, 3},
        torch::TensorOptions().dtype(torch::kByte).device(image.device()));

    dim3 threads(16, 16);
    dim3 blocks(cdiv(width, threads.x), cdiv(height, threads.y));
    blur_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        result.data_ptr<unsigned char>(),
        image.data_ptr<unsigned char>(),
        width, height, radius);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}