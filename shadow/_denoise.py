"""Denoising algorithms for post-demosaic linear RGB images.

Built-in:   none (denoising is always an optional enhancement)
Optional:
  shadow[denoise]     — BM3D (CPU, high quality)
  shadow[denoise-gpu] — bilateral / DnCNN / DRUNet via kornia + deepinv + PyTorch
"""
from __future__ import annotations

import math
from collections.abc import Callable
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


def count_tiles(H: int, W: int, tile_size: int) -> int:
    """Return the number of advance() calls denoise_image will make for DnCNN/DRUNet.

    Matches the runtime branch in denoise_image exactly:
    - Image fits in one tile (H ≤ tile_size AND W ≤ tile_size) → 1
    - Otherwise → number of write-region tiles produced by _tile_denoise

    Used by callers to pre-compute the total operation count for progress bars.
    BM3D and BILATERAL always advance once regardless of image size.
    """
    if H <= tile_size and W <= tile_size:
        return 1
    overlap = max(32, tile_size // 8)
    step = tile_size - overlap
    return math.ceil(H / step) * math.ceil(W / step)


def _tile_denoise(
    model_fn,
    t: "torch.Tensor",
    tile: int,
    overlap: int,
    on_tile_done: "Callable[[], None] | None" = None,
) -> "torch.Tensor":
    """Run a denoising model in overlapping tiles to stay within VRAM limits.

    Write regions are non-overlapping strides of (tile - overlap), guaranteeing
    every pixel is written exactly once with no gaps. Each write region is extended
    outward by overlap//2 pixels on each side (clamped at image edges) to give the
    model boundary context and prevent seam artefacts.

    model_fn:     callable (patch: Tensor[1,C,H,W]) → Tensor[1,C,H,W]
    t:            input tensor, shape (1, C, H, W), on the target device
    tile:         spatial size of each square tile (pixels)
    overlap:      context strip added on each side of the write region (pixels, even)
    on_tile_done: called once per completed tile (for progress tracking)
    """
    import torch

    _done = on_tile_done or (lambda: None)
    _, C, H, W = t.shape
    step = tile - overlap
    half = overlap // 2

    # Accumulate on CPU regardless of the inference device.  Non-contiguous
    # slice assignment into a device tensor (e.g. DirectML on AMD/Intel) can
    # silently fail for certain index patterns, leaving strips of zeros.
    # Reading patches FROM the device tensor is fine; only writes are unsafe.
    out = torch.zeros(1, C, H, W, dtype=torch.float32)

    for wy0 in range(0, H, step):
        wy1 = min(wy0 + step, H)
        ty0, ty1 = max(0, wy0 - half), min(H, wy1 + half)
        ry0, ry1 = wy0 - ty0, wy0 - ty0 + (wy1 - wy0)

        for wx0 in range(0, W, step):
            wx1 = min(wx0 + step, W)
            tx0, tx1 = max(0, wx0 - half), min(W, wx1 + half)
            rx0, rx1 = wx0 - tx0, wx0 - tx0 + (wx1 - wx0)

            patch = t[:, :, ty0:ty1, tx0:tx1]
            p_out = model_fn(patch)
            out[:, :, wy0:wy1, wx0:wx1] = p_out[:, :, ry0:ry1, rx0:rx1].cpu()
            _done()

    return out  # CPU tensor; caller does .cpu().numpy() which is then a no-op


def denoise_image(
    rgb: np.ndarray,
    kernel: DenoiseKernel,
    sigma: float = 0.05,
    tile_size: int = 512,
    on_advance: "Callable[[int], None] | None" = None,
) -> np.ndarray:
    """Denoise a float32 (H, W, 3) linear RGB image in [0, 1].

    rgb:        float32 array, shape (H, W, 3), values in [0, 1].
    kernel:     algorithm to use; see DenoiseKernel.
    sigma:      noise strength estimate. Range 0.02 (subtle) to 0.15 (heavy).
                Meaning: sigma_psd for BM3D; colour/space sigma for BILATERAL;
                noise std dev for DNCNN / DRUNET. Default 0.05.
    tile_size:  spatial tile size for DnCNN / DRUNet (default 512). Reduce to
                256 or 128 if you run out of VRAM. Ignored by BM3D / BILATERAL.
    on_advance: optional progress callback — called with advance count (1 per
                tile for DnCNN/DRUNet, once for BM3D/BILATERAL when done).

    Returns float32 (H, W, 3) with the same value range.
    """
    _adv = on_advance if on_advance is not None else lambda n: None

    if kernel == DenoiseKernel.BM3D:
        try:
            import bm3d
        except ImportError as exc:
            raise ImportError(
                "DenoiseKernel.BM3D requires bm3d: pip install 'shadow[denoise]'"
            ) from exc
        result = bm3d.bm3d(rgb, sigma_psd=sigma).astype(np.float32)
        _adv(1)
        return result

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
        _adv(1)
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

        with torch.no_grad():
            if H <= tile_size and W <= tile_size:
                result = model_fn(t)
                _adv(1)
            else:
                result = _tile_denoise(
                    model_fn, t,
                    tile=tile_size, overlap=overlap,
                    on_tile_done=lambda: _adv(1),
                )

        return result[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)

    raise ValueError(f"Unknown denoise kernel: {kernel!r}")  # pragma: no cover
