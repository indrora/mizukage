# Shader-Based and Non-Neural Denoising for shadow

This document explores GPU shader-based and classical spatial denoisers —
particularly [glslSmartDeNoise](https://github.com/BrutPitt/glslSmartDeNoise)
and the filter collection at [lygia.xyz](https://lygia.xyz) — as alternatives
to the neural-network kernels in `shadow/_denoise.py`, and discusses how the
per-pixel sigma values derivable from the L16's VST calibration data
(see `calibration-data-guide.md §1`) can be exploited in these approaches.

---

## Why consider non-neural denoisers?

The neural kernels in shadow (BM3D, bilateral, DnCNN, DRUNet) share a common
weakness for L16 images: they were trained or tuned on synthetic AWGN or
natural-image noise distributions that do not match the L16's noise profile.
The L16's dominant noise sources are:

- **Shot noise** — Poisson-distributed, signal-dependent.  Variance ∝ signal.
- **Read noise** — roughly Gaussian, signal-independent.
- **Hot pixels** — fixed-pattern; must be masked before any filter is applied.
- **Colour channel imbalance** — vignetting and CRA effects vary spatially.

The VST noise model in `calibration.lri` gives `a` and `b` coefficients per
channel so that `sigma(x,y) = sqrt(a * raw(x,y) + b)` is accurate per pixel.
Classical spatial denoisers can consume this spatially-varying sigma directly,
without the training-distribution mismatch that makes neural kernels sub-optimal.

Classical filters also have zero inference overhead, run on any GPU or CPU, and
need no model download.

---

## glslSmartDeNoise

**Repo:** https://github.com/BrutPitt/glslSmartDeNoise  
**License:** BSD 2-clause

glslSmartDeNoise is a single-file GLSL bilateral/smart denoise shader that
computes a weighted average of neighbouring pixels where the weight kernel
combines:

1. A **spatial Gaussian** of radius `uSigma` (pixels).
2. An **intensity Gaussian** of threshold `uThreshold` (colour distance).

The filter is sometimes described as "edge-preserving bilateral filter with
a specular/brightness-aware kernel" — it down-weights neighbours that differ
significantly in luminance, preserving edges and fine texture.

### Per-pixel sigma integration

The key insight: `uThreshold` controls how aggressively the filter smooths
across colour boundaries.  In the standard use case, a scalar threshold is
used for the whole image.  With the VST model, we can supply a **sigma
texture** — a single-channel float image where each texel holds
`sigma(x,y) = sqrt(a * raw_green(x,y) + b)` — and scale `uThreshold`
locally:

```glsl
// Modified kernel weight (pseudo-code)
float sigma_local = texture(sigmaTex, uv).r;
float intensity_dist = length(center_color - neighbor_color);
float w = spatial_gauss * exp(-0.5 * pow(intensity_dist / sigma_local, 2.0));
```

This makes the filter more aggressive in underexposed (noisy) regions and
more conservative in bright (clean) regions — exactly the right behaviour
for shot-noise-dominated sensors.

### Python integration path

glslSmartDeNoise is a GLSL fragment shader.  The integration options are:

**Option A — ModernGL (no display required):**

```python
import moderngl
import numpy as np

def glsl_smart_denoise(rgb: np.ndarray, sigma_map: np.ndarray,
                        u_sigma: float = 3.0) -> np.ndarray:
    """
    rgb:       float32 (H, W, 3) in [0, 1]
    sigma_map: float32 (H, W) per-pixel noise sigma from VST model
    u_sigma:   spatial kernel radius in pixels
    """
    H, W = rgb.shape[:2]
    ctx = moderngl.create_standalone_context()
    # ... compile shader, upload textures, render to FBO, read back
    # Full implementation: ~50 lines of moderngl boilerplate
    ...
```

`moderngl` runs headless on any OpenGL 3.3+ driver (Windows ANGLE,
Linux Mesa, macOS, or NVIDIA/AMD via their native drivers).  No display
server needed.

**Option B — via Shadertoy / offline renderer:**  
Export the image and sigma map, run the shader in RenderDoc, Shadertoy, or
a custom viewer.  Useful for one-off inspection but not for batch processing.

**Option C — reimplement in numpy/scipy:**  
The glslSmartDeNoise kernel is mathematically identical to a bilateral filter
with a per-pixel-varying range sigma.  It can be approximated in pure Python:

```python
from scipy.ndimage import gaussian_filter

def smart_denoise_numpy(rgb, sigma_map, spatial_sigma=3.0, n_iter=1):
    """Bilateral filter with spatially-varying range sigma."""
    # Simple approximation: iterate joint-bilateral passes
    # Not GPU-accelerated; use for correctness testing only
    ...
```

For production use, Option A (modernGL) or the kornia bilateral with a
sigma-map extension (see §4 below) is recommended.

---

## lygia.xyz

**Site:** https://lygia.xyz  
**Repo:** https://github.com/patriciogonzalezvivo/lygia  
**License:** LYGIA Software License (free for non-commercial; commercial license required)

lygia is a GLSL/HLSL/WGSL shader library with a broad collection of image
processing primitives in `lygia/filter/`.  The most relevant for L16 post-
processing:

### lygia/filter/denoise

| Shader | Algorithm | Notes |
|--------|-----------|-------|
| `denoise/bilateral.glsl` | Bilateral filter | Range + spatial weights; standard variant |
| `denoise/kuwahara.glsl` | Kuwahara filter | Anisotropic; good for painterly output, less for fidelity |
| `denoise/median.glsl` | Median filter | Good for salt-and-pepper / hot pixels |
| `denoise/nlm.glsl` | Non-local means | High quality; expensive (O(patch_radius²)) |
| `denoise/aces.glsl` | ACES-aware spatial filter | Tone-mapping aware; useful post-CCM |

### lygia/filter/sharpen and lygia/filter/chromaAbberration

After denoising, the L16 images can benefit from:

- `sharpen/unsharp.glsl` — classical unsharp mask after denoising.
- `sharpen/cas.glsl` — AMD Contrast Adaptive Sharpening; handles edge halos
  better than unsharp mask after aggressive denoising.
- `chromaAberration.glsl` — lateral chromatic aberration correction using a
  radial warp; overlaps with the distortion correction from the geometry
  calibration but can handle residual fringing after CCM.

### Per-pixel sigma in lygia filters

The `bilateral.glsl` implementation in lygia accepts a scalar range sigma.
Adapting it to a sigma map is a one-line change:

```glsl
// Original:
float sigmaDomain = uSigma;

// With VST sigma map:
float sigmaDomain = texture(u_sigmaTex, st).r;
```

The sigma texture is pre-computed from the VST model coefficients and the
raw green-channel data, matching the spatial resolution of the output image.

### Integration via modernGL or Pillow-SIMD

lygia shaders include each other via `#include` with relative paths, which
requires a small preprocessor to inline dependencies.  The lygia repo
ships a Python/Node.js resolver:

```python
from lygia import resolve  # pip install lygia (unofficial; or vendor the files)

shader_src = resolve("lygia/filter/denoise/bilateral.glsl")
# Then compile with moderngl as above
```

Alternatively, copy the relevant `.glsl` files into the project and inline
manually — the bilateral filter is ~80 lines.

---

## Constructing the per-pixel sigma map

From `calibration-data-guide.md §1`:

```python
import numpy as np

def build_sigma_map(raw_green: np.ndarray,
                    analog_gain: float,
                    vst_entries: list[dict]) -> np.ndarray:
    """
    raw_green:   uint16 (H, W) green channel from Bayer, pre-black-subtraction
    analog_gain: float from CaptureMetadata (e.g. 3.875)
    vst_entries: list of dicts with keys gain_x100, green: {a, b}

    Returns float32 (H, W) per-pixel sigma in normalised [0,1] units.
    """
    # Find nearest gain entry
    gain_code = round(analog_gain * 100 / 25) * 25
    entry = next(
        (e for e in vst_entries if e["gain_x100"] == gain_code),
        vst_entries[-1],  # fallback to max-gain entry
    )
    a = entry["green"]["a"]
    b = entry["green"]["b"]

    # Variance in raw counts, then convert to [0,1] normalised units
    white = 1023 - 42  # white_level - black_level
    variance_raw = a * raw_green.astype(np.float32) + b
    sigma_raw = np.sqrt(np.maximum(variance_raw, 0.0))
    return (sigma_raw / white).astype(np.float32)
```

The resulting sigma map:
- Is **larger** in shadow regions (high noise relative to signal).
- Is **smaller** in highlights (shot noise is proportionally smaller, and
  the sensor approaches saturation clipping, not noise).
- Varies per-channel (R/G/B have slightly different `a` and `b`).

For the bilateral filter, using the **green channel** sigma as a proxy for
all three channels is a reasonable approximation; the full treatment would
build a per-channel sigma and apply it to each colour plane independently.

---

## Hot pixels and spatial filters

**Critical ordering:** the hot-pixel bitmap from `hotpixel.rec` must be
applied **before** any spatial filter.  A spatial filter will spread hot-pixel
energy into neighbouring pixels, making correction after the fact impossible
without artefacts.  See `calibration-data-guide.md §5` for the bitmap format.

```python
# Correct order:
# 1. Subtract black level
# 2. Apply hot-pixel mask (interpolate same-channel Bayer neighbours)
# 3. Build sigma map from corrected green channel
# 4. Apply spatial denoiser with sigma map
# 5. Demosaic
# 6. Vignetting correction
# 7. CCM (forward matrix)
# 8. Gamma / tone mapping
```

---

## Comparison: spatial vs. neural denoisers for L16

| Property | glslSmartDeNoise / lygia bilateral | BM3D | DRUNet / DnCNN |
|----------|-----------------------------------|------|----------------|
| Per-pixel sigma input | ✓ (with minor shader edit) | Partial (BM3D-vsm) | ✗ (scalar sigma only) |
| Hot-pixel awareness | ✗ (pre-mask required) | ✗ | ✗ |
| L16 noise model match | ✓ (via VST coefficients) | Partial | Poor (AWGN-trained) |
| GPU acceleration | ✓ (GLSL/shader) | ✗ (CPU only) | ✓ (PyTorch) |
| Dependencies | modernGL or similar | bm3d package | torch + deepinv |
| Inference time (4K) | < 100 ms | 30–120 s | 5–30 s |
| Output quality | Good (detail retention) | Very good | Good–Very good |
| Edge preservation | Good (bilateral) | Excellent | Good |

For L16 images at typical capture gains (1–4×), the spatial bilateral
approach with the VST sigma map is likely to **outperform** generic neural
denoisers because the calibration-matched sigma gives the filter accurate
per-pixel smoothing targets.  Neural denoisers start to win only at very
high gains (≥ 4×) where the noise becomes complex enough to benefit from
learned priors.

---

## Recommended approach

1. **Always first:** hot-pixel masking (Bayer-domain, same-channel interpolation).
2. **Build sigma map** from VST calibration and the raw green channel.
3. **Spatial bilateral pass** (glslSmartDeNoise or lygia bilateral) with the
   sigma map as the range kernel.  Fast, no dependencies beyond an OpenGL
   context (modernGL) or a numpy approximation.
4. **Optional refinement:** if the spatial pass leaves structured noise visible
   at high ISO, follow with DRUNet at a low sigma (0.02–0.03) to handle
   residual correlations the bilateral pass cannot remove.

### Packaging

Since the spatial path needs only modernGL (or a pure-numpy fallback), it
fits well as a new optional extra:

```toml
# pyproject.toml
denoise-spatial = ["moderngl>=5.10"]
```

with a pure-numpy fallback that requires no extra install, making it the
lowest-friction denoising option in the shadow pipeline.
