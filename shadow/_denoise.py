"""Denoising algorithms for post-demosaic linear RGB images.

Built-in:   none (denoising is always an optional enhancement)
Optional:
  shadow[denoise]     — BM3D (CPU, high quality)
  shadow[denoise-gpu] — bilateral / DnCNN / DRUNet via kornia + deepinv + PyTorch
"""
from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np


class DenoiseKernel(str, Enum):
    """Denoising algorithm selector.

    Requires ``pip install shadow[denoise]``:
      BM3D      — Block Matching 3D; gold-standard classical CPU denoiser.
                  Typical sigma: 0.02 (subtle) – 0.15 (heavy).

    Requires ``pip install shadow[denoise-gpu]``:
      BILATERAL — Edge-preserving bilateral filter via kornia + PyTorch.
                  Fast, no model download. 10-50x quicker than BM3D on GPU.
      DNCNN     — Deep CNN blind denoiser (deepinv). Downloads ~3 MB weights
                  on first use. GPU-accelerated; good speed/quality balance.
      DRUNET    — Deeper residual U-Net (deepinv). Downloads ~37 MB weights on
                  first use. Best quality of the three GPU options; sigma-aware.

    All GPU kernels run on CUDA (NVIDIA/AMD ROCm), DirectML (AMD/Intel Windows),
    MPS (Apple Silicon), or CPU — whichever is available first.
    """

    BM3D      = "bm3d"
    BILATERAL = "bilateral"
    DNCNN     = "dncnn"
    DRUNET    = "drunet"


# Module-level model cache: key = "name:device_type"
_model_cache: dict[str, Any] = {}


def _best_torch_device():
    """Return the best available torch device across all GPU vendors.

    Priority: CUDA/ROCm → DirectML (AMD/Intel Windows) → MPS (Apple) → CPU.
    torch.cuda.is_available() is True for both NVIDIA CUDA and AMD ROCm builds
    of PyTorch, so no separate ROCm branch is needed.
    torch-directml is detected opportunistically; it need not be a declared dep.
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


def _deepinv_model(name: str, device) -> Any:
    """Load and cache a deepinv denoising model.

    Models are downloaded on first use (~3–37 MB depending on model) and
    cached in memory for the process lifetime to avoid reloading on each export.
    """
    cache_key = f"{name}:{device}"
    if cache_key not in _model_cache:
        try:
            import deepinv as dinv
        except ImportError as exc:
            raise ImportError(
                f"DenoiseKernel.{name.upper()} requires deepinv: "
                "pip install 'shadow[denoise-gpu]'"
            ) from exc

        if name == "dncnn":
            model = dinv.models.DnCNN(
                in_channels=3,
                out_channels=3,
                pretrained="download",
                device=device,
            )
        elif name == "drunet":
            model = dinv.models.DRUNet(
                in_channels=3,
                out_channels=3,
                pretrained="download",
                device=device,
            )
        else:  # pragma: no cover
            raise ValueError(f"Unknown deepinv model: {name!r}")

        _model_cache[cache_key] = model.eval()
    return _model_cache[cache_key]


def denoise_image(
    rgb: np.ndarray,
    kernel: DenoiseKernel,
    sigma: float = 0.05,
) -> np.ndarray:
    """Denoise a float32 (H, W, 3) linear RGB image in [0, 1].

    rgb:    float32 array, shape (H, W, 3), values in [0, 1].
    kernel: algorithm to use; see DenoiseKernel.
    sigma:  noise strength estimate. Range 0.02 (subtle) to 0.15 (heavy).
            Meaning: sigma_psd for BM3D; colour/space sigma for BILATERAL;
            noise std dev for DNCNN / DRUNET. Default 0.05.

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
        t = torch.from_numpy(rgb.transpose(2, 0, 1)[np.newaxis]).to(device)
        ks = _bilateral_ksize(sigma)
        result = KF.bilateral_blur(t, (ks, ks), sigma, (sigma * 10, sigma * 10))
        return result[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)

    if kernel in (DenoiseKernel.DNCNN, DenoiseKernel.DRUNET):
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                f"DenoiseKernel.{kernel.value.upper()} requires torch: "
                "pip install 'shadow[denoise-gpu]'"
            ) from exc

        device = _best_torch_device()
        model = _deepinv_model(kernel.value, device)
        t = torch.from_numpy(rgb.transpose(2, 0, 1)[np.newaxis]).to(device)
        with torch.no_grad():
            result = model(t, sigma=sigma)
        return result[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)

    raise ValueError(f"Unknown denoise kernel: {kernel!r}")  # pragma: no cover
