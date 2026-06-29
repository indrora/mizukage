# Using L16 Calibration Data in Image Processing

This document maps each dataset in `calibration-data-guide.md` to concrete
processing applications, notes what `shadow` already does, and identifies the
gaps where calibration data is available but not yet applied.

---

## What shadow already uses

| Calibration data | Where used | Status |
|-----------------|-----------|--------|
| `black_level` (42) | `_image.py` `to_raw_numpy()` — subtracted per pixel | ✓ used |
| `white_level` (1023) | `_image.py` normalisation to [0, 1] | ✓ used |
| AWB gains from ViewPreferences | `_image.py` `to_debayered_numpy()` per-channel Bayer multiply | ✓ used |
| Forward matrix (D65) | `_image.py` `_apply_forward_matrix()` — sensor RGB → XYZ D50 → sRGB | ✓ used |
| Device orientation (IMU) | `_image.py` `_orient()` — 180° + hold correction | ✓ used |
| Sensor model / camera ID | metadata / `info` display | ✓ used |

Everything below is **available in the calibration files but not yet applied**.

---

## 1. Denoising applications

### 1.1 Hot-pixel masking (highest priority)

**What:** `hotpixel.rec` contains a 4160×3120 boolean bitmap per camera marking
pixels characterised as hot at gain=7.75.

**Current state:** parsed and counted by `shadow calib`, but never applied to
image data.

**Application — in-Bayer, before demosaic:**

```python
def apply_hot_pixel_mask(bayer: np.ndarray, bitmap: np.ndarray,
                          r_row: int, r_col: int) -> np.ndarray:
    """Replace hot pixels with the mean of same-colour Bayer neighbours."""
    result = bayer.copy()
    hot_coords = np.argwhere(bitmap == 1)
    H, W = bayer.shape
    for row, col in hot_coords:
        # Determine colour channel of this pixel
        ch_row = row % 2
        ch_col = col % 2
        # Collect neighbours of the same colour (±2 rows/cols, same parity)
        neighbours = []
        for dr in (-2, 2):
            nr = row + dr
            if 0 <= nr < H and nr % 2 == ch_row:
                neighbours.append(bayer[nr, col])
        for dc in (-2, 2):
            nc = col + dc
            if 0 <= nc < W and nc % 2 == ch_col:
                neighbours.append(bayer[row, nc])
        if neighbours:
            result[row, col] = np.mean(neighbours)
    return result
```

This must run **before** any neural denoiser. Networks trained on natural images
will attempt to preserve hot pixels as fine detail.

The bitmap is conservative — characterised at worst-case gain (7.75). At typical
capture gains (1–4), most flagged pixels are actually clean. Applying the mask
unconditionally is still safe: the interpolation error on a ~60–90% hot-pixel
bitmap at max gain is imperceptible because the neighbours are good.

### 1.2 VST per-pixel sigma (quality improvement for neural denoisers)

**What:** the `vst_model` in `sensor_data` provides a linear noise-variance model
for 28 gain steps (100–775, step 25), separately for R, G, B, and panchromatic:

```
variance(signal) = a * signal + b
sigma(pixel)     = sqrt(a * raw_value + b)
```

**Current state:** `shadow[denoise]` passes a single scalar sigma to BM3D /
bilateral / DnCNN / DRUNet. The scalar is user-supplied with no connection to
the factory characterisation.

**Better approach — spatially-varying sigma map:**

```python
def vst_sigma_map(raw_bayer: np.ndarray,
                  analog_gain: float,
                  vst_table: list[dict]) -> np.ndarray:
    """Compute per-pixel noise sigma from the factory VST model."""
    gain_code = round(analog_gain * 100 / 25) * 25
    gain_code = max(100, min(775, gain_code))
    # Find the nearest entry
    entry = min(vst_table, key=lambda e: abs(e["gain_x100"] - gain_code))
    a = entry["green"]["a"]
    b = entry["green"]["b"]
    # raw_bayer is pre-black-subtraction uint16
    sigma = np.sqrt(np.maximum(a * raw_bayer.astype(np.float32) + b, 0))
    return sigma  # shape (H, W), same as bayer
```

After demosaicing to (H, W, 3), the per-channel sigma maps can be averaged into
a single (H, W) map and used as:

- **FFDNet sigma map input** — FFDNet explicitly accepts a per-pixel sigma tensor
  instead of a scalar; see `denoising-networks.md §1.1`.
- **BM3D sigma_psd** — BM3D also accepts a per-pixel `sigma_psd` array (2D).
  This replaces the user-supplied `--denoise-sigma` flag with calibration-derived
  values, eliminating guesswork for the user.
- **Adaptive bilateral** — spatial sigma (sigma_space) can be scaled by the local
  sigma value to smooth more aggressively where the sensor is noisier.

**Integration point in shadow:** `_export_rgb8()` in `_image.py` calls
`denoise_image()` after normalisation. The VST map must be computed from the
pre-black-subtracted raw Bayer before normalisation, then resampled to match the
output resolution. The `RawImage` object has `analog_gain` and `_black_level`;
the VST table would need to be passed in from the calibration file.

---

## 2. Colour correction applications

### 2.1 Illuminant-adaptive CCM selection

**Current state:** `shadow` always selects the D65 forward matrix, falling back
to whichever illuminant is first if D65 is absent. The AWB gains are used to
balance channels but the matrix is not changed.

**Better approach:** match the CCM to the actual illuminant by comparing the
capture's AWB R/G and B/G ratios against the calibrated `rg_ratio`/`bg_ratio`
reference points for each illuminant:

```python
def best_illuminant_profile(color_profiles, awb_gains):
    """Return the ColorProfile whose neutral point is closest to the AWB gains."""
    if awb_gains is None:
        return next((p for p in color_profiles if p.illuminant == Illuminant.D65), None)
    # AWB gains are applied as (R*r_gain, G*1, B*b_gain).
    # The neutral point (grey) for illuminant I has rg = 1/r_gain, bg = 1/b_gain.
    captured_rg = 1.0 / awb_gains.r  if awb_gains.r  else 1.0
    captured_bg = 1.0 / awb_gains.b  if awb_gains.b  else 1.0
    def dist(p):
        return (p.rg_ratio - captured_rg)**2 + (p.bg_ratio - captured_bg)**2
    return min(color_profiles, key=dist)
```

The gain is largest under tungsten (AWB R/B gains are very high); D65 is
appropriate for daylight. For mixed or unknown illuminants, interpolating between
the two nearest CCMs (`t * FM_A + (1-t) * FM_D65`) gives a smoother result than
snapping to a single entry.

### 2.2 Per-camera CCM normalisation for multi-camera composites

**What:** the 16 L16 cameras are physically distinct modules with independent
factory CCMs. Even after applying each camera's own forward matrix, residual
colour differences remain because the calibration was done at fixed conditions.

**`relative_brightness`** from the vignetting block normalises each camera's
output so that equal raw values from different cameras produce equal luminance
after exposure normalisation:

```python
corrected_exposure = raw_exposure / vignetting.relative_brightness
```

The C-array cameras are noticeably brighter (C1 = 1.62, ~2/3 stop) because their
longer focal length concentrates more light per unit aperture.

**Cross-camera colour matching:** for any composite that blends data from
multiple modules (super-resolution, stereo, Gaussian splatting training images),
all cameras should be normalised to a common reference. Use the reference camera's
(typically B4) `rg_ratio`/`bg_ratio` as the target neutral point and derive
per-camera gain trims:

```python
ref_profile = color_profiles["B4"]["D65"]
for cam, profile in color_profiles.items():
    r_trim = ref_profile.rg_ratio / profile.rg_ratio
    b_trim = ref_profile.bg_ratio / profile.bg_ratio
    # Apply (r_trim, 1.0, b_trim) on top of the camera's own AWB gains
```

### 2.3 Flash white balance

**What:** `device_calibration` holds the CCT of the warm and cool LEDs. When
flash was fired, the scene illuminant is a mixture of ambient and flash.

**Application:** if flash was active at capture, a two-illuminant AWB model can
blend the flash CCM (derived from the LED CCT) with the ambient CCM:

```python
# Approximate flash illuminant as blackbody at ledcool_cct / ledwarm_cct
# Mix ratio from flash power fraction (not directly available in metadata)
flash_rg = blackbody_rg_ratio(device_cal["ledcool_cct"])
ambient_rg = awb_gains.r  # from ViewPreferences
```

This is speculative without flash power metadata, but the CCT values are
available for whenever that information becomes accessible.

---

## 3. Geometric correction applications

### 3.1 Distortion undistortion

**What:** each camera has a 5-coefficient Brown–Conrady radial distortion model
normalised to 4160 px. The coefficients are in the calibration geometry bundle.

**Current state:** not applied anywhere in shadow.

**Application — before any spatial processing:**

```python
import cv2

def undistort_image(img: np.ndarray, K: dict, distortion: dict) -> np.ndarray:
    fx = K["fx"]
    cx, cy = K["cx"], K["cy"]
    camera_matrix = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]], dtype=np.float64)
    k1, k2, _, _, k3 = distortion["radial_coeffs"]
    # OpenCV distortion vector: (k1, k2, p1, p2, k3) — L16 has no tangential
    dist_coeffs = np.array([k1, k2, 0.0, 0.0, k3], dtype=np.float64)
    return cv2.undistort(img, camera_matrix, dist_coeffs)
```

This is necessary for any stereo/3D reconstruction but optional for single-image
display (L16 distortion is mild for the B and C arrays; A-array 28 mm shows more
barrel distortion).

### 3.2 Vignetting correction

**What:** per-camera 17×13 multiplicative correction grid.

**Current state:** not applied.

**Application:**

```python
from scipy.ndimage import map_coordinates

def apply_vignetting(channel: np.ndarray, vig_grid: np.ndarray) -> np.ndarray:
    """channel: (H, W) float32. vig_grid: (13, 17) float32."""
    H, W = channel.shape
    gh, gw = vig_grid.shape                    # 13, 17
    ys = np.linspace(0, gh - 1, H)
    xs = np.linspace(0, gw - 1, W)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    factors = map_coordinates(vig_grid, [yy.ravel(), xx.ravel()],
                               order=1).reshape(H, W)
    return channel * factors
```

Apply per-channel on the demosaiced float image before CCM. For focus-dependent
cameras (C-array, 4 hall-code entries), select the nearest entry by lens position.

---

## 4. Recommended processing order

The calibration data informs a strict ordering. Each step depends on the previous
one being correct:

```
raw Bayer (uint16, full 4160×3120)
  │
  ├─ 1. Hot-pixel mask (hotpixel.rec)          ← in Bayer, before anything
  │     Replace masked pixels with same-colour neighbours
  │
  ├─ 2. Black level subtraction (black=42)      ← already done by shadow
  │
  ├─ 3. VST sigma map computation               ← compute NOW from raw values
  │     sigma(x,y) = sqrt(a * raw(x,y) + b)    before normalisation changes values
  │
  ├─ 4. Vignetting correction (17×13 grid)      ← per-channel, before demosaic
  │
  ├─ 5. AWB gain application                    ← already done by shadow
  │
  ├─ 6. Demosaic (Bayer → RGB)                  ← already done by shadow
  │
  ├─ 7. Normalise to [0, 1]                     ← already done by shadow
  │
  ├─ 8. Denoising with VST sigma map            ← improved: per-pixel sigma
  │     (currently: fixed scalar sigma)
  │
  ├─ 9. CCM / forward matrix                    ← already done by shadow
  │     (improvement: illuminant selection from AWB ratios)
  │
  ├─ 10. Exposure adjustment, gamma, orientation ← already done by shadow
  │
  └─ 11. Distortion undistortion (optional)     ← not yet done; needed for 3D
```

Steps 1, 3, 4, and 8 (hot-pixel mask, VST sigma, vignetting, improved denoising)
represent the highest-value additions not yet implemented.

---

## 5. Priority ranking for implementation

| Priority | Feature | Calibration source | Expected impact |
|----------|---------|-------------------|----------------|
| 1 | Hot-pixel masking | `hotpixel.rec` | Eliminates fixed-pattern artefacts before demosaic |
| 2 | Vignetting correction | `calibration.lri` vignetting | Removes corner darkening, critical for composites |
| 3 | VST sigma map for denoiser | `sensor_data.vst_model` | Per-pixel optimal denoising vs. guessed scalar |
| 4 | Illuminant-adaptive CCM | `module_calibration[].color` | Correct colours under tungsten/fluorescent |
| 5 | Relative brightness normalisation | `vignetting.relative_brightness` | Required for cross-camera composites to match |
| 6 | Distortion correction | `module_calibration[].geometry` | Needed for stereo, 3D reconstruction, SR |
