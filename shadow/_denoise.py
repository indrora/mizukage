"""Denoising algorithms for post-demosaic linear RGB images.

Built-in:   BILATERAL_SPATIAL (pure numpy, no extra deps)
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

    No extra dependencies:
      BILATERAL_SPATIAL — Spatial bilateral filter (pure numpy).
                  Edge-preserving; no GPU required. Slow on large images
                  (~10-30 s for 4160×3120) but uses no extra dependencies.
                  Typical sigma: 0.02 (subtle) – 0.15 (heavy).

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

    BILATERAL_SPATIAL = "bilateral_spatial"
    BM3D      = "bm3d"
    BILATERAL = "bilateral"
    DNCNN     = "dncnn"
    DRUNET    = "drunet"


# Type alias for user-supplied denoising functions.
# Signature: (rgb: float32 H×W×3 in [0,1], sigma: float) → float32 H×W×3 in [0,1].
# Passed directly to denoise_image() or make_tiled_denoiser().
DenoiseFn = Callable[[np.ndarray, float], np.ndarray]


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


def _bilateral_filter(
    img: np.ndarray,
    sigma_r: float,
    sigma_s: float = 3.0,
    radius: int = 5,
) -> np.ndarray:
    """Per-channel bilateral filter on a float32 (H, W, C) image.

    Iterates over a (2*radius+1)^2 window of pixel offsets and accumulates
    weighted contributions using both spatial (Gaussian sigma_s) and range
    (Gaussian sigma_r) weights.  This is O(N * (2*radius+1)^2) — 121 passes
    over the full image for radius=5 — but requires only numpy and no GPU.

    img:     float32 array, shape (H, W, C), values in [0, 1].
    sigma_r: range sigma — controls how strongly intensity differences suppress
             the weight.  Matches the outer sigma argument to denoise_image().
    sigma_s: spatial sigma — controls how far the spatial Gaussian reaches.
    radius:  half-width of the search window (pixels in each direction).
    """
    H, W, C = img.shape
    result = np.zeros_like(img)
    weight_sum = np.zeros((H, W, 1), dtype=np.float32)

    inv_2sr2 = -1.0 / (2.0 * sigma_r ** 2)
    inv_2ss2 = -1.0 / (2.0 * sigma_s ** 2)

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            spatial_w = np.exp((dy * dy + dx * dx) * inv_2ss2)
            # Skip negligible spatial weights to avoid polluting the sum.
            if spatial_w < 1e-6:
                continue

            # Shift the image by (dy, dx) with zero-padding at borders so that
            # rolled-in values from the opposite edge don't influence the result.
            shifted = np.roll(np.roll(img, dy, axis=0), dx, axis=1)
            if dy > 0:
                shifted[:dy, :, :] = 0.0
            elif dy < 0:
                shifted[dy:, :, :] = 0.0
            if dx > 0:
                shifted[:, :dx, :] = 0.0
            elif dx < 0:
                shifted[:, dx:, :] = 0.0

            # Range weight: sum of squared per-channel differences.
            range_diff = np.sum((img - shifted) ** 2, axis=2, keepdims=True)
            range_w = np.exp(range_diff * inv_2sr2)

            w = spatial_w * range_w          # shape (H, W, 1) — broadcast over C
            result += w * shifted
            weight_sum += w

    # Guard against a near-zero weight sum at heavily-padded border corners.
    return result / np.maximum(weight_sum, 1e-8)


def _denoise_bilateral_spatial(
    image: np.ndarray,
    *,
    sigma: float,
    on_advance: "Callable[[int], None] | None" = None,
) -> np.ndarray:
    """Spatial bilateral filter.  No extra dependencies beyond numpy.

    sigma_s is derived from sigma to give a spatially consistent window
    relative to the noise level; radius is capped at 8 to keep run-time
    tractable on full-resolution (4160×3120) images (~10-30 s on CPU).
    """
    _adv = on_advance if on_advance is not None else lambda n: None
    # Scale spatial sigma with range sigma; floor at 1.5 to avoid a trivial
    # single-pixel window for very small sigma values.
    sigma_s = max(1.5, sigma * 20.0)
    # Cap radius at 8 so the (2*r+1)^2 = 289 passes remain tractable.
    radius = min(int(sigma_s * 2.5), 8)
    result = _bilateral_filter(image, sigma_r=sigma, sigma_s=sigma_s, radius=radius)
    _adv(1)
    return np.clip(result, 0.0, 1.0)


def denoise_image(
    rgb: np.ndarray,
    kernel: "DenoiseKernel | DenoiseFn",
    sigma: float = 0.05,
    tile_size: int = 512,
    on_advance: "Callable[[int], None] | None" = None,
) -> np.ndarray:
    """Denoise a float32 (H, W, 3) linear RGB image in [0, 1].

    rgb:        float32 array, shape (H, W, 3), values in [0, 1].
    kernel:     algorithm to use.  Either a DenoiseKernel enum value (built-in
                algorithm) or a DenoiseFn callable ``(rgb, sigma) → rgb``
                (arbitrary user-supplied denoiser, called directly with no tiling).
                Use make_tiled_denoiser() to wrap a torch model_fn that needs tiling.
    sigma:      noise strength estimate. Range 0.02 (subtle) to 0.15 (heavy).
                Meaning: sigma_psd for BM3D; colour/space sigma for BILATERAL;
                noise std dev for DNCNN / DRUNET. Passed as-is to DenoiseFn callables.
                Default 0.05.
    tile_size:  spatial tile size for DnCNN / DRUNet (default 512). Reduce to
                256 or 128 if you run out of VRAM. Ignored by BM3D / BILATERAL and
                by DenoiseFn callables (which handle their own tiling, if any).
    on_advance: optional progress callback — called with advance count (1 per
                tile for DnCNN/DRUNet, once for BM3D/BILATERAL/DenoiseFn when done).

    Returns float32 (H, W, 3) with the same value range.
    """
    _adv = on_advance if on_advance is not None else lambda n: None

    # User-supplied callable: dispatch directly, no tiling — the caller is
    # responsible for tiling if needed (see make_tiled_denoiser()).
    if callable(kernel) and not isinstance(kernel, DenoiseKernel):
        result = kernel(rgb, sigma)
        _adv(1)
        return result

    if kernel == DenoiseKernel.BILATERAL_SPATIAL:
        return _denoise_bilateral_spatial(rgb, sigma=sigma, on_advance=on_advance)

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


def make_tiled_denoiser(
    model_fn: "Callable",
    tile_size: int = 512,
    on_advance: "Callable[[], None] | None" = None,
) -> DenoiseFn:
    """Wrap a torch model callable with shadow's overlapping-tile loop.

    Returns a DenoiseFn ``(rgb: np.ndarray, sigma: float) → np.ndarray`` that
    can be passed directly as the ``denoise=`` parameter to ``to_png()``,
    ``to_tiff()``, or ``denoise_image()``.

    model_fn:   any callable ``(patch: Tensor[1,C,H,W]) → Tensor[1,C,H,W]``.
                The sigma argument from the outer DenoiseFn is NOT forwarded —
                if your model needs sigma, close over it before calling
                make_tiled_denoiser(), e.g.:
                  ``make_tiled_denoiser(lambda patch: model(patch, sigma=0.05))``
    tile_size:  spatial size of each square tile (pixels).  Reduce to 256/128
                if you run out of VRAM.  Default 512.
    on_advance: called once per completed tile (no argument).  Use this for
                progress tracking inside the tiled loop; the outer on_advance
                callback passed to denoise_image() is called only once (on
                completion of the whole image) by the DenoiseFn dispatch path.

    Torch is imported lazily inside the returned closure; the device is chosen
    at make_tiled_denoiser() call-time via _best_torch_device() so all tiles
    run on the same device.
    """
    import torch
    device = _best_torch_device()

    def _denoise(rgb: np.ndarray, sigma: float) -> np.ndarray:
        # sigma is accepted for interface compatibility but not forwarded;
        # callers that need sigma should close over it in model_fn.
        t = torch.from_numpy(rgb.transpose(2, 0, 1)[np.newaxis]).to(device)
        H, W = rgb.shape[:2]
        overlap = max(32, tile_size // 8)
        with torch.no_grad():
            if H <= tile_size and W <= tile_size:
                out = model_fn(t)
                if on_advance is not None:
                    on_advance()
            else:
                out = _tile_denoise(model_fn, t, tile_size, overlap, on_advance)
        return out[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)

    return _denoise
