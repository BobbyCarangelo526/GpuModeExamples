import os, sys
import torch
from torchvision.io import read_image, write_png

# make repo-root utils.py importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_cuda

HERE = os.path.dirname(os.path.abspath(__file__))

ext = load_cuda(
    kernel_path=os.path.join(HERE, "blur.cu"),
    cpp_source="torch::Tensor blur_image(torch::Tensor image, int radius);",
    functions=["blur_image"],
    name="blur_image",
)

img_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "Pocket.jpeg")
radius   = int(sys.argv[2]) if len(sys.argv) > 2 else 8

x = read_image(img_path).permute(1, 2, 0).contiguous().cuda()   # <-- contiguous
print("input :", tuple(x.shape), x.dtype, "mean", round(x.float().mean().item(), 2))

y = ext.blur_image(x, radius)
print("output:", tuple(y.shape), y.dtype, "mean", round(y.float().mean().item(), 2))

out = os.path.join(HERE, "output_pytorch.png")
write_png(y.permute(2, 0, 1).cpu(), out)
print("wrote", out)