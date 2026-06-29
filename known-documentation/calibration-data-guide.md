# Light L16 Calibration Data: Contents and Processing Guide

This document describes the calibration data produced by the Light factory pipeline
for the L16 camera, explains what each dataset means for image processing, and
defines a condensed per-camera processing record that a pipeline can use directly
without re-parsing the raw LELR files at capture time.

All data described here was inspected on calibration files for device
`c2482092395721d7:5bb53e30fc0cf05f` (Light L16, firmware 1.0.53531).

---

## Source files

| File | Contents |
|------|----------|
| `calibration.lri` | Geometry, distortion, colour CCM, vignetting, sensor characterisation |
| `asic_calib_v1.lri` | ASIC-level geometry (same intrinsics as calibration.lri in practice) |
| `zoom_calib_v0.lri` | Per-focus intrinsics (also same values on this device) |
| `hotpixel.rec` | Per-camera hot-pixel bitmaps at multiple gain/temperature conditions |

All four are LELR block streams (`shadow calib <dir> --json` parses them all).

---

## 1. Sensor characterisation (`SensorCharacterization`)

Source: `calibration.lri`, `sensor_data` field of any `LightHeader` block.

| Field | Value | Use |
|-------|-------|-----|
| `black_level` | 42 | Subtract from every raw pixel before any processing |
| `white_level` | 1023 | Normalise to [0, 1] as `(pixel − black) / (white − black)` |
| `cliff_slope` | 2.0 | Slope of the soft-clipping knee beyond white_level |
| `vst_model` | 28 entries (gain 100–775, step 25) | Per-gain noise model for denoising |

### VST noise model

Each `vst_model` entry covers one gain setting and contains, for each of four
channels (R, G, B, panchromatic), a linear noise-variance model:

```
variance(signal) = a * signal + b
```

where `signal` is the raw (pre-black-subtracted) pixel value.  The `threshold`
and `scale` fields relate to the VST (Variance-Stabilising Transform) that
converts the sensor signal to a domain where additive Gaussian denoisers apply.

**Processing use:** look up the entry with `gain` nearest to `analog_gain * 100`
(e.g. gain=387.5 → use entry 375 or 400), then apply:

```
sigma_pixel = sqrt(a * raw_value + b)
```

as the per-pixel noise standard deviation for BM3D or similar denoisers.
The `shadow[denoise]` pipeline currently uses a fixed sigma; using the VST
model per-pixel would be a quality improvement.

---

## 2. Per-camera geometry

Source: `calibration.lri` (primary), `asic_calib_v1.lri`, `zoom_calib_v0.lri`.

On this device all three files produce identical intrinsics for each camera.
Use `calibration.lri` as the canonical source.

### 2.1 Intrinsics (camera matrix)

Each camera has **3 focus bundles** at focus distances 818 mm, 1500 mm, and a
repeated 818 mm entry (the third appears to be a reference or duplicate).
The hall code (encoder position of the VCM/lens actuator) maps focus distance
to a physical lens position.

The camera matrix `K` is stored as a 3×3 upper-triangular matrix:

```
K = [[fx,  0, cx],
     [ 0, fy, cy],
     [ 0,  0,  1]]
```

For the L16, `fx == fy` (square pixels, no skew).  Nominal values:

| Array | Focal length (mm) | fx (pixels) | Sensor |
|-------|--------------------|-------------|--------|
| A (28 mm) | ~28 | ~3376 | AR1335 4160×3120 |
| B (70 mm) | ~70 | ~8284 | AR1335 4160×3120 |
| C (150 mm)| ~150 | ~18700 | AR1335 4160×3120 |

`cx`, `cy` are the principal point in pixels (near but not exactly at image
centre due to manufacturing tolerances).

**Focus distance interpolation:** to get intrinsics at an arbitrary focus
distance, linearly interpolate between the two calibrated bundles that bracket
it using their `focus_hall_code` values as the interpolation parameter.

### 2.2 Radial distortion

A 5-coefficient radial polynomial model (Brown–Conrady style):

```
r_d = r_u * (1 + k1*r̂² + k2*r̂⁴ + k3*r̂⁶ + k4*r̂⁸ + k5*r̂¹⁰)
```

where `r̂ = r_u / normalization` (normalization = 4160 px on the X axis for
all cameras), and distances are measured from `distortion_center` (≈ principal
point but independently fitted).

The `valid_roi` field marks the pixel region over which the fit was validated
(typically the full sensor minus an 8-pixel border).

**Undistortion:** use OpenCV `undistortPoints` or equivalent with `(k1, k2, 0, 0, k3)`
(the L16 polynomial skips the P1/P2 tangential terms).

### 2.3 Chief Ray Angle (CRA) model

The `cra` sub-message describes how the lens chief ray angle varies with
image-plane radius.  It is used for:

- **Vignetting correction** (light falls off at oblique angles)
- **Chromatic vignetting / lateral colour** (different wavelengths respond
  differently to oblique illumination)
- **Light-field / plenoptic reconstruction** in the Lumen pipeline

The CRA table has 100 sample points `(r_mm, angle_deg)` where `r_mm` is the
distance from the CRA centre in millimetres and `angle_deg` is the chief ray
angle.  Pixel size is 1.1 µm; `r_px * 0.0011 = r_mm`.

For most single-image pipelines the CRA model can be ignored; it matters most
for cross-camera stereo and computational refocus.

### 2.4 Mirror type

| Value | Meaning |
|-------|---------|
| `NONE` | Fixed lens, no moving mirror (A array: 28 mm) |
| `GLUED` | Mirror glued in place at manufacture (some B and C modules) |
| `MOVABLE` | Mirror is actuated (most B and C modules) |

Movable mirrors are used for optical image stabilisation and focus adjustment.
Fixed/glued mirrors have a single fixed focal length; movable mirrors require
the `mirror_position` from the capture metadata to correctly compute the
effective focal length.

---

## 3. Colour calibration

Source: `calibration.lri`, `module_calibration[].color` repeated field.

Each camera has colour calibration for **3 standard illuminants**:

| Illuminant | Type value | Description |
|------------|------------|-------------|
| D65 | 2 | Daylight (6500 K) — default for outdoor/sRGB output |
| A   | 0 | Incandescent / tungsten (2856 K) |
| F11 | 6 | Fluorescent (TL84, 4000 K) |

Note: cameras A2 and C6 on this device have **no colour calibration** in
the factory file.

### Forward matrix (sensor RGB → XYZ D50)

A 3×3 matrix `FM` that converts white-balanced sensor RGB into CIE XYZ
under D50 illumination (DNG Profile Connection Space).  Compose with the
Bradford D50→D65 adaptation to obtain display-referred sRGB:

```
XYZ_D50 = FM @ sensor_rgb
linear_sRGB = M_D50_to_sRGB @ XYZ_D50
```

where `M_D50_to_sRGB` is the standard matrix (already built into `shadow/_image.py`).

**Illuminant selection:** choose the entry whose `rg_ratio` / `bg_ratio`
reference point is closest to the capture's AWB gains to minimise colour error.
In practice, D65 is correct for daylight and is the default in `shadow`.

### Color matrix (XYZ → camera RGB)

The inverse direction `CM = FM⁻¹` (approximately).  Used for rendering intent
transforms and in DNG files as `ColorMatrix1` / `ColorMatrix2`.

### rg_ratio / bg_ratio

The R/G and B/G ratio of a perfectly neutral (grey) scene under this illuminant,
as measured by this specific camera.  Used to verify or override AWB:

```python
# Simple illuminant-based AWB override
r_gain = 1.0 / cc.rg_ratio
b_gain = 1.0 / cc.bg_ratio
```

---

## 4. Vignetting correction

Source: `calibration.lri`, `module_calibration[].vignetting`.

### Vignetting grid

A **17×13** floating-point grid of multiplicative correction factors
(one value per tile; bilinearly interpolate to get the factor at any pixel).
Values > 1.0 brighten corners to compensate for lens fall-off.

Multiple entries keyed by `hall_code` (lens position) may exist for cameras
with focus-dependent vignetting; on this device most cameras have only a
single entry at `hall_code = 0`.  C-array cameras have 4 entries.

**Application:**

```python
# vig_grid is a (13, 17) float32 array
# vig_factor(x, y) is bilinear interpolation of vig_grid at pixel (x, y)
corrected = raw_image * vig_factor(x, y)
```

Apply after black subtraction, before demosaicing (or equivalently per-channel
after demosaicing).

### Crosstalk correction

A second **17×13** grid (`crosstalk`) that corrects for inter-pixel optical
crosstalk.  The format is a packed array; its exact application order in the
Lumen pipeline is not documented, but it is likely applied before the vignetting
grid.

### Relative brightness

A scalar that normalises this camera's exposure relative to the reference
camera.  C-array cameras can be significantly brighter (C1 `relative_brightness`
= 1.62) because their longer focal length concentrates more light.

---

## 5. Hot-pixel map

Source: `hotpixel.rec`.

Each camera has one measurement at gain=7.75 (gain code 775), short exposure,
at ambient temperature.  The bitmap is a flat `width × height` array of bytes
(0 = normal, 1 = hot) compressed with zlib, preceded by a 20-byte header:

```
[0:4]  uint32 LE  Unix timestamp
[4:8]  uint32     padding (0)
[8:12] uint32     compressed size
[12:16] uint32    width  (4160)
[16:20] uint32    height (3120)
[20:]              zlib-compressed boolean bitmap
```

Hot-pixel fractions at this characterisation gain:

| Array | Fraction hot |
|-------|-------------|
| A (28 mm) | 56–65% |
| B (70 mm) | 74–81% |
| C (150 mm) | 87–91% |

These fractions are **expected and normal** for maximum-gain factory
characterisation.  At typical capture gains (analog_gain ≈ 1–4) the effective
hot-pixel set is far smaller — the factory deliberately over-characterises
at worst-case gain.

**Processing:** the hot-pixel map at gain 7.75 serves as a conservative mask.
Apply before demosaicing:

```python
mask = (hp_bitmap == 1)   # True where hot
# Replace hot pixels with bilinear interpolation of Bayer neighbours
# (replace only within the same colour channel)
```

For captures at lower gain, the map may over-correct (mark pixels as hot that
are actually fine at lower gain).  Without a per-gain capture a safe heuristic
is to always apply the mask; the interpolation error for a small number of
incorrectly-masked pixels is negligible.

---

## 6. Device calibration

Source: `calibration.lri`, `device_calibration`.

Flash calibration for the two LEDs:

| Field | Unit | Description |
|-------|------|-------------|
| `ledcool_lux` | lux | Illuminance of the cool LED at 1 m |
| `ledcool_max_lumens` | lm | Maximum output of cool LED |
| `ledcool_cct` | K | Correlated colour temperature of cool LED |
| `ledwarm_lux` | lux | Illuminance of the warm LED at 1 m |
| `ledwarm_max_lumens` | lm | Maximum output of warm LED |
| `ledwarm_cct` | K | Correlated colour temperature of warm LED |

Used for mixed-illuminant white balance when flash was fired.

---

## Condensed processing record format

The following JSON structure captures everything a single-image processing
pipeline needs for one camera module.  It is designed to be pre-computed
once per device and stored alongside captures, eliminating repeated parsing
of the raw LELR files.

```json
{
  "device_id": "c2482092395721d7:5bb53e30fc0cf05f",
  "sensor": {
    "type": "AR1335",
    "black_level": 42,
    "white_level": 1023,
    "width_px": 4160,
    "height_px": 3120,
    "pixel_size_mm": 0.0011
  },
  "noise_model": [
    {
      "gain_x100": 100,
      "red":   {"a": 0.000195, "b": -7.24e-6},
      "green": {"a": 0.000193, "b": -7.27e-6},
      "blue":  {"a": 0.000194, "b": -7.28e-6}
    }
    // ... 28 entries total, gain_x100 100..775 step 25
  ],
  "cameras": {
    "B4": {
      "array": "B",
      "nominal_focal_length_mm": 70,
      "mirror_type": "GLUED",
      "focus_bundles": [
        {
          "focus_distance_mm": 818,
          "hall_code": 1608,
          "K": {
            "fx": 8284.24, "fy": 8284.24,
            "cx": 2073.86, "cy": 1551.42
          },
          "distortion": {
            "center_px": [2020.0, 1567.0],
            "normalization_px": 4160.0,
            "radial_coeffs": [0.02068, -0.06591, 0.0, 0.0, 0.03363],
            "valid_roi": [8, 8, 4143, 3103]
          }
        },
        {
          "focus_distance_mm": 1500,
          "hall_code": 1518,
          "K": {
            "fx": 8284.24, "fy": 8284.24,
            "cx": 2073.86, "cy": 1551.42
          },
          "distortion": { "...": "same structure" }
        }
      ],
      "color": {
        "D65": {
          "rg_ratio": 0.4947,
          "bg_ratio": 0.6496,
          "forward_matrix_3x3": [
            [0.8603, 0.1642, -0.0603],
            [0.2880, 1.0709, -0.3589],
            [-0.0652, -0.2289, 1.4052]
          ],
          "color_matrix_3x3": [
            [0.7389, -0.2290, -0.0650],
            [-0.2880, 1.1240, 0.1642],
            [0.0334, -0.0540, 0.4448]
          ]
        },
        "A":   { "...": "same structure" },
        "F11": { "...": "same structure" }
      },
      "vignetting": {
        "grid_w": 17, "grid_h": 13,
        "relative_brightness": 0.9911,
        "entries": [
          {
            "hall_code": 0,
            "factors": [ /* 221 floats, row-major, top-left to bottom-right */ ]
          }
        ]
      },
      "hot_pixels": {
        "gain_x100": 775,
        "timestamp": 1503014400,
        "count": 9876543,
        "fraction": 0.759,
        "bitmap": "<base64-encoded zlib-compressed bytes, or path to sidecar file>"
      }
    }
    // ... 15 more cameras
  }
}
```

### Field notes

**Interpolating intrinsics at capture time:**

```python
def interpolate_K(bundles, hall_code_at_capture):
    # bundles sorted by hall_code
    b0, b1 = bracket(bundles, hall_code_at_capture)
    t = (hall_code_at_capture - b0.hall_code) / (b1.hall_code - b0.hall_code)
    fx = lerp(b0.K.fx, b1.K.fx, t)
    cx = lerp(b0.K.cx, b1.K.cx, t)
    cy = lerp(b0.K.cy, b1.K.cy, t)
    return K(fx, fx, cx, cy)
```

The hall code at capture is in `CaptureMetadata.modules[cam].lens_position`.

**Vignetting application:**

```python
# Grid covers the full sensor; bilinear interpolation:
grid_x = pixel_x / (width  - 1) * (grid_w - 1)
grid_y = pixel_y / (height - 1) * (grid_h - 1)
factor = bilinear_interp(vig_grid, grid_x, grid_y)
corrected_pixel = raw_pixel * factor
```

**Noise sigma lookup at capture gain:**

```python
gain_code = round(analog_gain * 100 / 25) * 25   # snap to nearest entry
entry = noise_model[gain_code]
sigma = sqrt(entry["green"]["a"] * raw_value + entry["green"]["b"])
```

**Hot pixel correction (in-Bayer, before demosaic):**

```python
for row, col in hot_pixel_coordinates:
    # sample 4 neighbours of the same colour channel
    bayer[row, col] = mean_of_same_colour_neighbours(bayer, row, col)
```

---

## What is NOT in the calibration data

- **Extrinsics** (rotation/translation between cameras): present in the proto
  (`CalibrationFocusBundle.extrinsics`) but empty in this device's files.
  The Lumen pipeline likely derives inter-camera geometry at runtime from the
  mirror actuator state.
- **Per-temperature intrinsics**: `sensor_temp` is recorded in each focus bundle
  but the current device has only one temperature sample per bundle.
- **Per-gain hot-pixel maps**: only the single max-gain (775) map is present.
  Finer-grained characterisation at lower gains would allow more precise masking.
- **Temporal noise / dark current**: the VST model captures photon shot noise
  and read noise but not dark current accumulation at long exposures.
