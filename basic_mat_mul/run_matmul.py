import os, sys
import torch
from torchvision.io import read_image, write_png

# make repo-root utils.py importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_cuda

HERE = os.path.dirname(os.path.abspath(__file__))

ext = load_cuda(
    kernel_path=os.path.join(HERE, "mat_mul.cu"),
    cpp_source="torch::Tensor mat_mul(torch::Tensor tensor_1, torch::Tensor tensor_2);",
    functions=["mat_mul"],
    name="mat_mul",
)

M = 4
N = 2
K = 6

max_value = 4

tensor_1 = torch.randint(0, max_value, (M, K))
tensor_2 = torch.randint(0, max_value, (K, N))

print(f"tensor_1:\n{tensor_1}")
print(f"tensor_2:\n{tensor_2}")

torch_result = tensor_1.matmul(tensor_2) 
print(f"pytorch result shape: {torch_result.shape}, pytorch result:\n{torch_result}")

output = ext.mat_mul(tensor_1, tensor_2)
print(f"output: shape: {tuple(output.shape)}, values:\n {output}")
