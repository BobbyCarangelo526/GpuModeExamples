"""Shared helpers for GPUMODE / PMPP CUDA exercises."""
from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline
import matplotlib.pyplot as plt


def load_cuda(kernel_path, cpp_source, functions, name=None, verbose=True):
    """Compile a .cu file as a PyTorch extension and return the module.

    `functions` must match the host function name(s) in the .cu file.
    """
    cuda_source = Path(kernel_path).read_text()
    return load_inline(
        name=name or Path(kernel_path).stem,
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=functions,
        with_cuda=True,
        extra_cuda_cflags=["-O2"],
        verbose=verbose,
    )


def show(image_tensor, title="output"):
    """Display an HxW (grayscale) or HxWx3 (RGB) uint8 CUDA tensor."""
    img = image_tensor.detach().cpu().numpy()
    plt.figure(figsize=(10, 8))
    if img.ndim == 2:
        plt.imshow(img, cmap="gray", vmin=0, vmax=255)  # pin scale, no auto-normalize
    else:
        plt.imshow(img)
    plt.axis("off")
    plt.title(title)
    plt.show()
