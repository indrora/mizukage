# Denoising Networks for shadow: Alternatives and Arbitrary Model Loading

This document surveys denoising architectures that could integrate with
shadow's existing pipeline, notes their practical trade-offs for L16 images,
and proposes how to let users load arbitrary models — including those hosted on
HuggingFace — without modifying the library.

---

## The existing pipeline contract

`denoise_image()` in `shadow/_denoise.py` operates on **float32 (H, W, 3) in
[0, 1]** and returns the same shape.  Internally, torch-based kernels are
dispatched through a shared tile loop (`_tile_denoise`) that accepts any
`model_fn: Tensor[1,C,H,W] → Tensor[1,C,H,W]`.  Adding a new kernel means
either:

1. Adding a `DenoiseKernel` enum value and a branch in `denoise_image()`, or
2. Exposing `_tile_denoise` (or a wrapper) directly, so callers supply their
   own `model_fn` without touching the enum.

Option 2 is the right direction for arbitrary model loading — see §3.

---

## 1. Notable denoising architectures

### 1.1 FFDNet (Zhang et al., TIP 2018)

A fast CNN denoiser that accepts a noise-level map (per-pixel sigma) as input,
making it directly compatible with the VST noise model described in
`calibration-data-guide.md`.  Sigma-awareness is its key advantage over blind
denoisers: you can feed the per-gain `sqrt(a·raw + b)` value from the
calibration data as a spatially-varying sigma map rather than a scalar.

- **Pretrained weights:** available via Zhang Kai's KAIR repo on GitHub (gray
  and color models, ~2 MB each).  Not on HuggingFace Hub, but loadable via
  `torch.hub` or direct download.
- **Integration effort:** medium — model architecture needs to be copied or
  imported; weights loaded from a local checkpoint.  The tile loop in shadow
  already handles spatial tiling; the sigma map would need to be downscaled to
  match each tile.
- **Fit for L16:** strong — sigma-awareness lets you exploit the factory
  calibration's VST model precisely.

### 1.2 DRUNet (Zhang et al., CVPR 2021)

Already integrated via deepinv.  Listed here for completeness: it is also
sigma-aware, accepts a scalar sigma, and produces excellent results.  The
deepinv version auto-downloads weights.

### 1.3 SCUNet (Swin-Conv U-Net, Zhang et al., CVPR 2022)

Designed specifically for **real-world noise** (JPEG, compression, sensor
pattern noise) rather than AWGN.  Relevant because the L16 hot-pixel and
read-noise patterns differ from the synthetic Gaussian distributions that
DnCNN/DRUNet were trained on.

- **Pretrained weights:** KAIR repo (GitHub) and some HuggingFace community
  repos (search `SCUNet denoising`), typically ~30–60 MB.
- **Integration effort:** medium — architecture is a standard PyTorch
  `nn.Module`; weights loadable from `.pth`.  Blind denoiser (no sigma input),
  so `sigma` parameter would be ignored.
- **Fit for L16:** potentially best of the CNN family for real-world noise.

### 1.4 NAFNet (Chen et al., ECCV 2022)

Nonlinear Activation Free Network.  SOTA on SIDD and DND benchmarks as of its
publication; uses simple gating instead of activation functions.  Architecture
is compact and fast on GPU.

- **Pretrained weights:** `https://huggingface.co/rwightman/pytorch-image-models`
  does not carry NAFNet, but the official repo (megvii-research/NAFNet) provides
  `.pth` checkpoints, and community re-uploads exist on HuggingFace Hub.
- **Integration effort:** low once weights are obtained — plain `nn.Module`.
- **Fit for L16:** good for Gaussian noise removal; less tuned for hot-pixel
  patterns than SCUNet.

### 1.5 Restormer (Zamir et al., CVPR 2022)

Transformer-based image restoration.  Strong on Gaussian and real-noise removal
tasks.  Memory-hungry (full attention over large spatial windows); **requires
tiling for 4160×3120 images** even on 24 GB VRAM.

- **Pretrained weights:** official repo (swz30/Restormer) has `.pth` files;
  some HuggingFace community repos exist.
- **Integration effort:** medium — architecture module needed; tiling already
  handled by shadow's `_tile_denoise`.
- **Fit for L16:** likely excellent quality; VRAM cost is significant.

### 1.6 SwinIR (Liang et al., ICCV 2021)

Swin Transformer for image restoration.  Has a dedicated denoising variant
(`SwinIR-S`/`SwinIR-L` for Gaussian denoising at sigma 15/25/50) and a
real-noise variant.  Officially on HuggingFace:

```
caidas/swin2SR-realworld-sr-x4-64   # super-resolution, not denoising
```

Denoising weights are in the JingyunLiang/SwinIR GitHub releases; community
HuggingFace re-uploads exist (search `SwinIR denoising`).

- **Integration effort:** low — `from models.network_swinir import SwinIR` or
  use the HuggingFace `transformers` / `timm` path; tiling via shadow's loop.
- **Fit for L16:** good Gaussian denoising; `sigma` parameter maps directly.

### 1.7 Diffusion-based denoisers (DiffPIR, DPIR)

Score-based or DDPM denoisers treat denoising as approximate posterior sampling.
The deepinv library already exposes some of these.  Quality is very high but
inference is slow (many forward passes per image) and VRAM usage is large.
Practical for offline archival processing, not interactive use.

---

## 2. Self-supervised denoisers (no clean training data needed)

Standard supervised denoisers are trained on pairs of (noisy, clean) images.
The L16's noise distribution — dominated by the calibrated hot-pixel map and
the VST read-noise model — is unusual enough that fine-tuned or self-supervised
approaches could outperform generic pretrained weights.

### 2.1 Noise2Fast (Fang et al., 2022)

Trains on a **single noisy image at inference time**.  No clean reference
needed.  Converges in ~30 s on a mid-range GPU.  Results are competitive with
supervised denoisers when the noise model is poorly matched.

- **HuggingFace:** no canonical repo, but the paper code is short and
  self-contained; could be wrapped as a callable directly.

### 2.2 Noise2Self / Noise2Void

Blind-spot denoisers.  Slightly lower quality than supervised but useful when
the noise statistics diverge from training distributions.

### 2.3 Per-pixel sigma from VST + classical denoiser

Rather than a neural network at all: use the VST calibration data
(`calibration-data-guide.md §1`) to compute `sigma(x,y)` per pixel and pass
it to a spatially-adaptive filter (FFDNet, guided filter, or nlmeans with a
varying h parameter).  This exploits the factory characterisation more directly
than any generic pretrained network.

---

## 3. Proposed API for arbitrary model loading

### 3.1 Accept a callable as `denoise`

The simplest change: widen the `denoise` parameter in `_export_rgb8()`,
`to_png()`, `to_tiff()`, and `denoise_image()` to accept either a
`DenoiseKernel` enum value **or a plain callable**.

```python
# Type alias (add to shadow/_denoise.py or shadow/_types.py)
type DenoiseFn = Callable[[np.ndarray, float], np.ndarray]
# (rgb: float32 H×W×3, sigma: float) → float32 H×W×3

# Widened signature (no enum changes needed)
def denoise_image(
    rgb: np.ndarray,
    kernel: DenoiseKernel | DenoiseFn,
    sigma: float = 0.05,
    tile_size: int = 512,
    on_advance: Callable[[int], None] | None = None,
) -> np.ndarray:
    ...
    if callable(kernel) and not isinstance(kernel, DenoiseKernel):
        # User-supplied function: call directly, no tiling
        result = kernel(rgb, sigma)
        _adv(1)
        return result
    ...
```

For models that need tiling (Restormer, SwinIR), the user wraps them to accept
`(H, W, 3)` numpy and handles tiling themselves, or shadow exposes a helper:

```python
def make_tiled_denoiser(
    model_fn: Callable,   # Tensor[1,C,H,W] → Tensor[1,C,H,W]
    tile_size: int = 512,
    on_advance: Callable[[], None] | None = None,
) -> DenoiseFn:
    """Wrap a torch model_fn with shadow's overlapping-tile loop."""
    import torch
    device = _best_torch_device()

    def _denoise(rgb: np.ndarray, sigma: float) -> np.ndarray:
        t = torch.from_numpy(rgb.transpose(2, 0, 1)[np.newaxis]).to(device)
        H, W = rgb.shape[:2]
        overlap = max(32, tile_size // 8)
        fn = lambda patch: model_fn(patch)  # noqa: E731
        with torch.no_grad():
            if H <= tile_size and W <= tile_size:
                out = fn(t)
                if on_advance:
                    on_advance()
            else:
                out = _tile_denoise(fn, t, tile_size, overlap, on_advance)
        return out[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)

    return _denoise
```

### 3.2 HuggingFace loader helper

A `load_hf_denoiser()` function that handles the three common HuggingFace
model shapes:

```python
def load_hf_denoiser(
    repo_id: str,
    *,
    filename: str | None = None,   # specific .pth/.safetensors file in the repo
    model_cls=None,                # nn.Module class if architecture must be supplied
    tile_size: int = 512,
    device=None,
) -> DenoiseFn:
    """Load a denoising model from HuggingFace Hub and return a DenoiseFn.

    Three loading strategies are attempted in order:

    1. transformers.pipeline("image-to-image", model=repo_id) — works for
       models registered with the HuggingFace Transformers image-restoration
       pipeline (currently rare for denoisers, more common for SR).

    2. huggingface_hub.hf_hub_download(repo_id, filename) + torch.load() —
       downloads a raw .pth checkpoint.  Requires model_cls to be supplied
       (the architecture is not stored in the file).

    3. torch.hub.load(repo_id, ...) — for models on GitHub that declare a
       hubconf.py (e.g. NAFNet, SCUNet from their official repos).

    Requires: pip install huggingface_hub (always), transformers (strategy 1),
              torch (strategies 2–3).
    """
    import torch
    dev = device or _best_torch_device()

    # Strategy 1: transformers pipeline
    try:
        from transformers import pipeline as hf_pipeline
        pipe = hf_pipeline("image-to-image", model=repo_id, device=dev)

        def _hf_pipe_denoise(rgb: np.ndarray, sigma: float) -> np.ndarray:
            from PIL import Image as PILImage
            uint8 = (rgb * 255).clip(0, 255).astype(np.uint8)
            pil_in = PILImage.fromarray(uint8)
            pil_out = pipe(pil_in)[0]["generated_image"]  # transformers ≥4.40
            return np.asarray(pil_out).astype(np.float32) / 255.0

        return _hf_pipe_denoise
    except Exception:
        pass

    # Strategy 2: raw checkpoint + supplied architecture
    if model_cls is not None:
        from huggingface_hub import hf_hub_download
        fname = filename or "model.pth"
        ckpt_path = hf_hub_download(repo_id=repo_id, filename=fname)
        state = torch.load(ckpt_path, map_location="cpu")
        # Checkpoints may wrap weights under a 'params' or 'state_dict' key
        state_dict = state.get("params", state.get("state_dict", state))
        model = model_cls()
        model.load_state_dict(state_dict, strict=False)
        model = model.eval().to(dev)

        return make_tiled_denoiser(
            lambda patch: model(patch),
            tile_size=tile_size,
        )

    raise ValueError(
        f"Could not load {repo_id!r} automatically.  "
        "Supply model_cls= for a raw checkpoint, or ensure the repo uses "
        "the transformers image-to-image pipeline."
    )
```

### 3.3 Usage examples

**SwinIR denoising (sigma=25 checkpoint from JingyunLiang/SwinIR):**

```python
# Download the architecture once and point at a community HF upload
from network_swinir import SwinIR  # from the official SwinIR repo

denoiser = shadow._denoise.load_hf_denoiser(
    "eugenesiow/SwinIR-denoising",   # hypothetical community upload
    filename="005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth",
    model_cls=lambda: SwinIR(upscale=1, img_size=128, window_size=8,
                              img_range=1.0, depths=[6]*6, embed_dim=180,
                              num_heads=[6]*6, mlp_ratio=2,
                              upsampler='', resi_connection='1conv'),
    tile_size=256,
)

lri = shadow.open_lri("photo.lri")
lri.reference_image.to_png("out.png", denoise=denoiser, denoise_sigma=25/255)
```

**FFDNet with VST per-pixel sigma (future integration):**

```python
# FFDNet accepts a sigma map tensor of shape (1,1,H,W) or (1,H,W,1) depending
# on implementation.  A thin adapter is needed:
def ffdnet_with_vst(ffdnet_model, vst_entry, raw_bayer):
    """Returns a DenoiseFn that uses per-pixel noise from the VST model."""
    a, b = vst_entry["green"]["a"], vst_entry["green"]["b"]

    def _denoise(rgb: np.ndarray, sigma_ignored: float) -> np.ndarray:
        import torch
        # Approximate raw green channel from debayered G
        g_approx = rgb[:, :, 1]
        sigma_map = np.sqrt(np.maximum(a * g_approx + b, 0)).astype(np.float32)
        t_rgb = torch.from_numpy(rgb.T[np.newaxis])      # (1,3,H,W)
        t_sig = torch.from_numpy(sigma_map[np.newaxis, np.newaxis])  # (1,1,H,W)
        with torch.no_grad():
            out = ffdnet_model(t_rgb, t_sig)
        return out[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)

    return _denoise
```

**Noise2Fast (self-supervised, no pretrained weights):**

```python
# noise2fast.py from paper code — trains on the image itself
def noise2fast_denoiser(n_iters: int = 2000) -> DenoiseFn:
    def _denoise(rgb: np.ndarray, sigma: float) -> np.ndarray:
        from noise2fast import denoise  # user installs from paper repo
        return denoise(rgb, n_iters=n_iters)
    return _denoise

img.to_png("out.png", denoise=noise2fast_denoiser(1000))
```

---

## 4. Dependency and packaging strategy

| Approach | New `pyproject.toml` extra | Required by user |
|----------|---------------------------|-----------------|
| Built-in FFDNet | `denoise-ffdnet = ["torch>=2.0"]` | weights path |
| HuggingFace loader | `denoise-hf = ["huggingface_hub>=0.20", "torch>=2.0"]` | repo_id, optionally model_cls |
| transformers pipeline | `denoise-hf = [..., "transformers>=4.40"]` | repo_id only |
| Arbitrary callable | no new dep | user supplies callable |

The arbitrary-callable path (§3.1) requires **zero new dependencies** and can
ship immediately.  The `load_hf_denoiser` helper can be added as an opt-in
extra.

---

## 5. Practical recommendation for L16 images

| Priority | Kernel | Why |
|----------|--------|-----|
| 1 | DRUNet (already in shadow) | Best quality for Gaussian noise at known sigma |
| 2 | SCUNet (load_hf_denoiser / custom load) | Better fit for L16 real-world noise patterns |
| 3 | FFDNet with VST sigma map | Uses factory calibration data per pixel; uniquely good fit |
| 4 | Noise2Fast | When noise model is unknown or calibration unavailable |
| 5 | Restormer / SwinIR | SOTA quality, high VRAM cost, tiling required |

The hot-pixel correction (apply mask before demosaic, interpolate from
same-channel neighbours) should always run **before** any neural denoiser.
Neural networks trained on natural images cannot reliably distinguish hot pixels
from fine detail and will attempt to preserve them.
