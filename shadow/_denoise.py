"""Denoising algorithms for post-demosaic linear RGB images.

Built-in:  none (denoising is always an optional enhancement)
Optional:  BM3D (``pip install shadow[denoise]``)
"""
from __future__ import annotations

from enum import Enum

import numpy as np


class DenoiseKernel(str, Enum):
    """Denoising algorithm selector.

    All algorithms require ``pip install shadow[denoise]`` (bm3d package).

      BM3D — Block Matching 3D; gold-standard classical denoiser. Accepts
             float32 (H, W, 3) in [0, 1] and a ``sigma`` noise estimate.
             Typical sigma: 0.02 (subtle) – 0.15 (heavy).
    """

    BM3D = "bm3d"


def denoise_image(
    rgb: np.ndarray,
    kernel: DenoiseKernel,
    sigma: float = 0.05,
) -> np.ndarray:
    """Denoise a float32 (H, W, 3) linear RGB image in [0, 1].

    rgb:    float32 array, shape (H, W, 3), values in [0, 1].
    kernel: algorithm to use; see DenoiseKernel.
    sigma:  noise standard deviation estimate (sigma_psd for BM3D).
            Typical range 0.02 (subtle) to 0.15 (heavy). Default 0.05.

    Returns float32 (H, W, 3) with the same value range.

    Requires ``pip install shadow[denoise]``.
    """
    if kernel == DenoiseKernel.BM3D:
        try:
            import bm3d
        except ImportError as exc:
            raise ImportError(
                "DenoiseKernel.BM3D requires bm3d: pip install 'shadow[denoise]'"
            ) from exc
        return bm3d.bm3d(rgb, sigma_psd=sigma).astype(np.float32)

    raise ValueError(f"Unknown denoise kernel: {kernel!r}")  # pragma: no cover
