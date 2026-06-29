# shadow: Next Steps and Project Ideas

Synthesised from five research documents generated across this session:
`calibration-data-guide`, `calibration-data-applications`, `denoising-networks`,
`shader-denoising`, `scene-reconstruction`.

---

## What is already implemented

| Feature | File |
|---------|------|
| LELR block parsing, all proto types | `shadow/_block.py`, `shadow/_proto.py` |
| 16-camera RAW unpacking (packed 10/12/14-bit, Bayer-JPEG) | `shadow/_unpack.py` |
| Bilinear demosaic; optional high-quality kernels | `shadow/_debayer.py` |
| AWB gains from ViewPreferences | `shadow/_proto.py` |
| D65 forward-matrix colour correction | `shadow/_image.py` |
| 180° sensor-mount fix + IMU orientation detection | `shadow/_image.py`, `shadow/_proto.py` |
| Black / white level from SensorData | `shadow/_proto.py` |
| BM3D, bilateral, DnCNN, DRUNet denoising | `shadow/_denoise.py` |
| `shadow info` (human + JSON), `shadow extract`, `shadow export` | `shadow/cli/commands/` |
| `shadow calib` — calibration directory dump (human + JSON) | `shadow/cli/commands/calib.py` |
| Hot-pixel statistics from `hotpixel.rec` | `shadow/cli/commands/calib.py` |
| LRIS depth-map reader | `shadow/_lris.py` |

---

## Near-term: calibration data integration

These all use data already parsed by `shadow calib`; none require new
dependencies beyond what is already in the project.

### 1. Hot-pixel masking (highest priority)

Every neural denoiser in the existing pipeline will try to preserve hot pixels
as fine detail. The factory bitmap must be applied **before demosaic** to be
effective.

- Parse `hotpixel.rec` at export time; decode the per-camera zlib bitmap.
- Replace hot pixels with the bilinear mean of the four same-channel Bayer
  neighbours (RGGB/BGGR: only R-neighbours for R pixels, etc.).
- Expose as `shadow export --hot-pixel-correct` (optional, opt-in, since the
  max-gain map over-masks at low gain).
- Requires: a calibration directory path (`--calib`) or an auto-discovered
  `lightcal/` directory beside the LRI.

### 2. Per-pixel VST sigma map

The factory VST model gives `σ(x,y) = sqrt(a·raw + b)` per pixel at each gain
setting. This is more accurate than any user-guessed scalar.

- At export time, look up the VST entry nearest `analog_gain × 100`.
- Compute the sigma map on the pre-normalised raw array (before black
  subtraction, since `a`/`b` are fit to raw counts).
- Use as input to BM3D (`sigma_psd` can be an array) or FFDNet (see §3.2).
- Short-term: expose as `--use-vst-sigma` flag on `shadow export`.

### 3. Illuminant-adaptive colour correction

The current pipeline always picks D65. The calibration file provides D65, A,
and F11 matrices with per-camera `rg_ratio`/`bg_ratio` neutral-point
references.

Heuristic: compute `rg_capture = awb_gains.r / awb_gains.gr` and
`bg_capture = awb_gains.b / awb_gains.gb`, then pick the illuminant whose
`(rg_ratio, bg_ratio)` is closest in Euclidean distance.

- Implementation: add to `_image._export_rgb8()` when a calibration context is
  available; otherwise keep the D65 default.

### 4. Vignetting correction

A 17×13 multiplicative grid per camera. Bilinear-interpolate to get the
correction factor at each pixel; apply per-channel after black subtraction,
before demosaic.

```python
grid_x = pixel_x / (width - 1) * 16   # 0..16
grid_y = pixel_y / (height - 1) * 12  # 0..12
factor = bilinear(vig_grid, grid_x, grid_y)
bayer[row, col] *= factor
```

C-array cameras have 4 hall-code-indexed entries; interpolate by
`lens_position` from `CaptureMetadata.modules`.

### 5. Relative brightness normalisation

When compositing images from different focal-length arrays, the
`relative_brightness` scalar (per camera, from vignetting calibration) should
normalise luminance before blending. C1 is ~1.62× brighter than the reference.

### 6. Lens distortion undistortion

Required for any stereo or 3D work; optional for single-image display.
The calibration provides a 5-coefficient radial polynomial and a distortion
centre per camera per focus bundle.

```python
# OpenCV mapping
cv2.undistort(image, K, dist_coeffs=(k1, k2, 0, 0, k3))
```

Interpolate `K` and distortion coefficients between focus bundles using
`lens_position` (hall code) from `CaptureMetadata.modules`.

---

## Near-term: denoising API improvements

### 7. Accept a callable as `denoise=`

Zero new dependencies. Widen `denoise_image()` and all `to_png`/`to_tiff`
signatures to accept `DenoiseKernel | Callable[[np.ndarray, float], np.ndarray]`.
A callable is invoked directly; enum values go through the existing branches.

This immediately unlocks all the architectures described in
`denoising-networks.md` without touching the enum.

### 8. `make_tiled_denoiser(model_fn, tile_size)` helper

Wraps any `Tensor[1,C,H,W] → Tensor[1,C,H,W]` torch callable with the
existing `_tile_denoise` loop. Eliminates boilerplate for Restormer, SwinIR,
FFDNet, SCUNet — any model that needs spatial tiling.

### 9. `load_hf_denoiser(repo_id, ...)` helper (new `denoise-hf` extra)

Three-strategy cascade:
1. `transformers.pipeline("image-to-image", model=repo_id)`
2. `hf_hub_download` + `torch.load` + supplied `model_cls`
3. `torch.hub.load`

Requires `pip install 'shadow[denoise-hf]'` → `huggingface_hub`, optionally
`transformers`.

---

## Medium-term: spatial / shader-based denoising

### 10. Calibration-matched spatial bilateral

At capture gains 1–4× (typical for well-lit scenes), a bilateral filter driven
by the VST per-pixel sigma is likely to **outperform generic neural denoisers**
because the sigma map gives it accurate per-pixel smoothing targets. Neural
denoisers start winning only above ~4× gain where the noise becomes complex
enough to require learned priors.

Options (ordered by integration friction):

| Option | Deps | Quality | Speed |
|--------|------|---------|-------|
| NumPy/SciPy spatial bilateral | none | medium | slow |
| glslSmartDeNoise via `moderngl` | `moderngl` | good | fast |
| LYGIA `denoise/bilateral.glsl` | `moderngl` | good | fast |
| LYGIA `denoise/nlm.glsl` | `moderngl` | high | medium |
| FFDNet with VST sigma map | `torch` | high | medium |

**The sigma-texture pipeline** (for any GL path):
1. Compute `σ_map = sqrt(a·raw_green + b) / (white - black)` on CPU.
2. Save as 16-bit PNG sidecar.
3. Load as `uniform sampler2D u_sigma` in the shader.
4. Use `σ = texture(u_sigma, uv).r` as the range-kernel standard deviation.

Proposed new extra: `denoise-spatial = ["moderngl>=5.10"]` with a pure-NumPy
fallback.

### 11. FFDNet with per-pixel VST sigma map

FFDNet accepts a full spatial sigma map as input — uniquely suited to exploit
the factory calibration. Weights are ~2 MB from the KAIR GitHub repo (not on
HuggingFace Hub; loadable via `torch.hub` or direct download).

---

## Medium-term: cross-camera pipeline

### 12. Per-camera calibration context at export time

`shadow export` currently processes one camera at a time with no calibration
awareness. Add an optional `--calib <dir>` argument that:
- Loads the condensed processing record (see `calibration-data-guide.md §7`).
- Applies per-camera hot-pixel mask, vignetting grid, and VST sigma for that
  specific camera ID.
- Uses illuminant-adaptive CCM based on AWB ratios.
- Applies `relative_brightness` normalisation.

### 13. Multi-camera composite export

Export all cameras in a single pass, normalised in luminance and colour, into a
directory or a multi-page TIFF. Useful as input to stereo and 3D pipelines.

---

## Long-term: stereo, depth, and 3D reconstruction

All 3D work is gated on recovering the inter-camera extrinsics. The
`CalibrationFocusBundle.extrinsics` proto field exists but is empty in this
device's calibration files. Four recovery approaches (see
`scene-reconstruction.md` for detail):

| Approach | Effort | Quality |
|----------|--------|---------|
| COLMAP SfM (K from calibration, distortion → OPENCV model) | medium | high on textured scenes |
| DUSt3R/MASt3R transformer pose estimation (same-array pairs) | low | good |
| Decode `MovableMirrorFormat.mirror_system` proto geometry | high (undocumented) | exact |
| Physical measurement + PnP refinement | very high | exact |

Extrinsics are fixed by the chassis — recover once per device, reuse for all
captures.

### 14. Stereo depth from same-array pairs

B1–B5 (70 mm) are the best-conditioned stereo pair due to the short baseline
and matched optics. After extrinsics recovery:
- Rectify the pair using calibration K + recovered R/t.
- Run SGM or RAFT-Stereo for disparity.
- Compare against / fuse with the LRIS disparity map (195×260).

### 15. 3D Gaussian Splatting

Maps cleanly to COLMAP input: K from calibration, R/t from SfM, undistorted
images from `shadow export`. Notes:
- Per-camera appearance conditioning is important — 16 cameras have subtly
  different colour responses even after CCM.
- `AngleOpticalCenterMapping` from the mirror geometry should correct
  mirror-induced principal-point shifts for precision work.

### 16. NeRF

Nerfacto (Nerfstudio) with per-image appearance embeddings absorbs residual
camera-to-camera colour variation. Zip-NeRF handles the wide focal-length
spread (28 / 70 / 150 mm) better than vanilla NeRF.

### 17. Sub-pixel super-resolution

The intra-array parallax (A1–A5 at 28 mm, with slight sub-pixel offsets) can
drive classical super-resolution. Requires sub-pixel registration between
cameras, which in turn requires extrinsics.

### 18. Self-supervised fine-tuning of denoisers

The L16's hot-pixel and read-noise patterns differ from synthetic Gaussian
distributions. Noise2Fast (trains on a single image, ~30 s on GPU) or
Noise2Void (blind-spot, no clean reference) can fine-tune or replace generic
pretrained denoisers when the noise model is poorly matched.

---

## Priority ranking

| # | Item | Effort | Impact |
|---|------|--------|--------|
| 1 | Hot-pixel correction before demosaic | low | high — required for all denoisers |
| 2 | Accept callable as `denoise=` | low | high — unlocks all external models |
| 3 | Per-pixel VST sigma map | low | high — improves all denoising paths |
| 4 | Vignetting correction | low | medium — visible in corners |
| 5 | Illuminant-adaptive CCM | low | medium — measurable colour accuracy |
| 6 | `make_tiled_denoiser` + `load_hf_denoiser` | medium | medium — enables SCUNet, FFDNet |
| 7 | glslSmartDeNoise via moderngl | medium | medium — best denoiser at low gain |
| 8 | `--calib` flag on `shadow export` | medium | high — ties everything together |
| 9 | Lens distortion undistortion | medium | required for 3D |
| 10 | Extrinsics recovery (COLMAP) | high | required for all 3D work |
| 11 | Stereo depth from B-array pairs | high | new capability |
| 12 | 3DGS / NeRF pipeline | very high | new capability |
