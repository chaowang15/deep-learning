"""Print the torch version and which accelerator is available (CUDA, Apple MPS, or CPU).

Run from code/diffusion/:  python scripts/check_env.py
"""

import torch


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"  # Apple Silicon GPU
    return "cpu"


if __name__ == "__main__":
    print("torch :", torch.__version__)
    device = pick_device()
    print("device:", device)
    if device == "cuda":
        print("gpu   :", torch.cuda.get_device_name(0))
    x = torch.randn(2, 3, device=device)
    print("ok    :", (x @ x.T).shape)
