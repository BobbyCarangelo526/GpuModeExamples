import os, sys
import torch
from torchvision.io import read_image, write_png

# make repo-root utils.py importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_cuda

HERE = os.path.dirname(os.path.abspath(__file__))

# ext = load_cuda(
#     kernel_path=os.path.join(HERE, "mat_mul.cu"),
#     cpp_source="torch::Tensor mat_mul(torch::Tensor image, torch::Tensor image);",
#     functions=["mat_mul"],
#     name="mat_mul",
# )

M = 4
N = 2
K = 10

tensor_1 = torch.randint(0, 10, (M, K))
tensor_2 = torch.randint(0, 10, (K, N))

print(f"tensor_1:\n{tensor_1}")
print(f"tensor_2:\n{tensor_2}")

torch_result = tensor_1 @ tensor_2
print(f"pytorch result:\n{tensor_2}")


# y = ext.blur_image(x, radius)
# print("output:", tuple(y.shape), y.dtype, "mean", round(y.float().mean().item(), 2))

# out = os.path.join(HERE, "output_pytorch.png")
# write_png(y.permute(2, 0, 1).cpu(), out)
# print("wrote", out)