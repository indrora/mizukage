"""Denoising algorithms for post-demosaic linear RGB images.

Built-in:   none (denoising is always an optional enhancement)
Optional:
  shadow[denoise]     — BM3D (CPU, high quality)
  shadow[denoise-gpu] — bilateral filter via kornia + PyTorch (GPU-accelerated)
"""
from __future__ import annotations

from enum import Enum

import numpy as np


class DenoiseKernel(str, Enum):
    """Denoising algorithm selector.

    Requires ``pip install shadow[denoise]``:
      BM3D      — Block Matching 3D; gold-standard classical CPU denoiser.
                  Accepts float32 (H, W, 3) in [0, 1] and a sigma estimate.
                  Typical sigma: 0.02 (subtle) – 0.15 (heavy).

    Requires ``pip install shadow[denoise-gpu]``:
      BILATERAL — Edge-preserving bilateral filter via kornia + PyTorch.
                  Runs on CUDA (NVIDIA/AMD ROCm), DirectML (AMD/Intel Windows),
                  MPS (Apple Silicon), or CPU in that priority order.
                  Typically 10-50x faster than BM3D when a GPU is present.
    """

    BM3D = "bm3d"
    BILATERAL = "bilateral"


def _best_torch_device():
    """Return the best available torch device across all GPU vendors.

    Priority: CUDA/ROCm → DirectML (AMD/Intel Windows) → MPS (Apple) → CPU.
    torch.cuda.is_available() is True for both NVIDIA CUDA and AMD ROCm builds
    of PyTorch, so no separate ROCm branch is needed.
    torch-directml is detected opportunistically and used automatically on
    Windows AMD/Intel without needing to be a declared dependency.
    """
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")           # NVIDIA CUDA or AMD ROCm
    try:
        import torch_directml                 # AMD/Intel/NVIDIA on Windows
        return torch_directml.device()
    except ImportError:
        pass
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")            # Apple Silicon
    return torch.device("cpu")


def _bilateral_ksize(sigma: float) -> int:
    """Odd kernel size for bilateral blur, scaled to sigma (range 5–21)."""
    k = max(5, int(sigma * 100) | 1)         # |1 forces odd
    return min(k, 21)


def denoise_image(
    rgb: np.ndarray,
    kernel: DenoiseKernel,
    sigma: float = 0.05,
) -> np.ndarray:
    """Denoise a float32 (H, W, 3) linear RGB image in [0, 1].

    rgb:    float32 array, shape (H, W, 3), values in [0, 1].
    kernel: algorithm to use; see DenoiseKernel.
    sigma:  noise strength (sigma_psd for BM3D; colour/space sigma for
            bilateral). Range 0.02 (subtle) to 0.15 (heavy). Default 0.05.

    Returns float32 (H, W, 3) with the same value range.
    """
    if kernel == DenoiseKernel.BM3D:
        try:
            import bm3d
        except ImportError as exc:
            raise ImportError(
                "DenoiseKernel.BM3D requires bm3d: pip install 'shadow[denoise]'"
            ) from exc
        return bm3d.bm3d(rgb, sigma_psd=sigma).astype(np.float32)

    if kernel == DenoiseKernel.BILATERAL:
        try:
            import torch
            import kornia.filters as KF
        except ImportError as exc:
            raise ImportError(
                "DenoiseKernel.BILATERAL requires kornia: "
                "pip install 'shadow[denoise-gpu]'"
            ) from exc

        device = _best_torch_device()
        # (H, W, 3) float32 → (1, 3, H, W) tensor on the chosen device
        t = torch.from_numpy(rgb.transpose(2, 0, 1)[np.newaxis]).to(device)
        ks = _bilateral_ksize(sigma)
        # sigma_color: how strongly colour differences gate smoothing
        # sigma_space: spatial reach in pixels (grows with sigma)
        result = KF.bilateral_blur(t, (ks, ks), sigma, (sigma * 10, sigma * 10))
        return result[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)

    raise ValueError(f"Unknown denoise kernel: {kernel!r}")  # pragma: no cover
