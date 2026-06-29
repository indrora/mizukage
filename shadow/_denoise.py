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


def _tile_denoise(model_fn, t: "torch.Tensor", tile: int, overlap: int) -> "torch.Tensor":
    """Run a denoising model in overlapping tiles to stay within VRAM limits.

    Tiles are generated with the given step (tile - overlap). For each tile
    the model runs on the full patch, but only the interior (excluding
    overlap//2 on each non-edge side) is written to the output to avoid
    seam artefacts from border effects in the model.

    model_fn: callable (patch: Tensor[1,C,H,W]) → Tensor[1,C,H,W]
    t:        input tensor, shape (1, C, H, W), on the target device
    tile:     spatial size of each square tile (pixels)
    overlap:  overlap strip width (pixels); must be even
    """
    import torch

    _, C, H, W = t.shape
    step = tile - overlap
    half = overlap // 2

    def _tile_starts(length: int) -> list[int]:
        if length <= tile:
            return [0]
        pts = list(range(0, length - tile, step))
        pts.append(length - tile)       # always include the trailing tile
        return sorted(set(pts))

    ys = _tile_starts(H)
    xs = _tile_starts(W)

    out = torch.zeros_like(t)

    for i, y0 in enumerate(ys):
        y1 = min(y0 + tile, H)
        for j, x0 in enumerate(xs):
            x1 = min(x0 + tile, W)

            patch = t[:, :, y0:y1, x0:x1]
            p_out = model_fn(patch)

            # Crop strip to keep: skip overlap/2 on all interior (non-edge) sides
            iy0 = half if i > 0 else 0
            ix0 = half if j > 0 else 0
            iy1 = (y1 - y0) - (half if i < len(ys) - 1 else 0)
            ix1 = (x1 - x0) - (half if j < len(xs) - 1 else 0)

            out[:, :, y0 + iy0:y0 + iy1, x0 + ix0:x0 + ix1] = \
                p_out[:, :, iy0:iy1, ix0:ix1]

    return out


def denoise_image(
    rgb: np.ndarray,
    kernel: DenoiseKernel,
    sigma: float = 0.05,
    tile_size: int = 512,
) -> np.ndarray:
    """Denoise a float32 (H, W, 3) linear RGB image in [0, 1].

    rgb:       float32 array, shape (H, W, 3), values in [0, 1].
    kernel:    algorithm to use; see DenoiseKernel.
    sigma:     noise strength estimate. Range 0.02 (subtle) to 0.15 (heavy).
               Meaning: sigma_psd for BM3D; colour/space sigma for BILATERAL;
               noise std dev for DNCNN / DRUNET. Default 0.05.
    tile_size: spatial tile size for DnCNN / DRUNet (default 512). Reduce to
               256 or 128 if you run out of VRAM. Ignored by BM3D / BILATERAL.

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

        H, W = rgb.shape[:2]
        overlap = max(32, tile_size // 8)
        model_fn = lambda patch: model(patch, sigma=sigma)   # noqa: E731

        if H <= tile_size and W <= tile_size:
            with torch.no_grad():
                result = model_fn(t)
        else:
            with torch.no_grad():
                result = _tile_denoise(model_fn, t, tile=tile_size, overlap=overlap)

        return result[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)

    raise ValueError(f"Unknown denoise kernel: {kernel!r}")  # pragma: no cover
