# mizukage

A Python library and CLI for reading Light L16 camera files — raw image data, depth maps, and factory calibration.

The Light L16 is a 16-lens computational camera. Each capture fires up to 16 independent sensors across three focal lengths (28 mm / 70 mm / 150 mm equivalent), producing a bundle of `.lri` raw files and an optional `.lris` depth-map sidecar. `mizukage` parses the proprietary LELR block-stream format, exposes sensor data as NumPy arrays, and can export processed images with full calibration applied.

---

## Installation

```bash
pip install mizukage
# or with uv:
uv add mizukage
```

Optional dependency groups:

| Group | Adds |
|-------|------|
| `mizukage[demosaic]` | Malvar 2004, Menon 2007, and DDFAPD high-quality debayering |
| `mizukage[denoise]` | BM3D denoising (CPU) |
| `mizukage[denoise-gpu]` | GPU bilateral, DnCNN, and DRUNet via kornia + deepinv + PyTorch |
| `mizukage[explorer]` | `mizukage calib-view` interactive calibration viewer (DearPyGui) |

---

## CLI

### `mizukage info`

Print metadata from an LRI or LRIS file.

```
mizukage info photo.lri
mizukage info photo.lri --cameras     # per-camera exposure, gain, AWB, focus
mizukage info photo.lri --blocks      # raw LELR block layout and sizes
mizukage info photo.lri --json        # machine-readable JSON
mizukage info photo.lris              # depth-map sidecar metadata
```

### `mizukage export`

Export each camera module's image from an LRI file.

```bash
# All cameras -> PNG, debayered + AWB + sRGB gamma
mizukage export photo.lri ./out

# Single camera, TIFF, +1 stop exposure
mizukage export photo.lri ./out --camera B4 --format tiff --exposure +1.0

# Raw 16-bit Bayer (no debayering)
mizukage export photo.lri ./out --raw

# High-quality demosaic (requires mizukage[demosaic])
mizukage export photo.lri ./out --kernel menon

# With factory calibration: hot-pixel correction + sigma-matched denoising
mizukage export photo.lri ./out --calib images/lightcal --denoise bm3d

# Full calibration pipeline: hot-pixel + vignetting + undistortion + denoising
mizukage export photo.lri ./out \
    --calib images/lightcal \
    --denoise bm3d \
    --undistort
```

Output filenames follow the pattern `A1.png`, `B4_raw.tiff`, etc.

#### Calibration pipeline (`--calib`)

When `--calib DIR` points at a `lightcal` directory, corrections are applied in order, all in linear light before gamma:

1. **Hot-pixel correction** — per-camera defect maps from `hotpixel.rec`
2. **Vignetting correction** — multiplicative falloff grid from `calibration.lri`
3. **Denoising** (if `--denoise`) — sigma is set automatically from the factory VST noise model
4. **CCM** — factory `forward_matrix` chosen by matching capture AWB gains to the nearest colour profile
5. **Lens undistortion** (if `--undistort`) — inverse-map radial polynomial from `calibration.lri`
6. **Gamma / tone mapping**

### `mizukage extract`

Dump raw Bayer data as NumPy `.npy` files (one `uint16` array per camera).

```bash
mizukage extract photo.lri ./raw
mizukage extract photo.lri ./raw --camera B4 --camera C1
mizukage extract photo.lri ./raw --no-subtract-black   # keep sensor black-level offset
```

A `metadata.json` summary is written alongside the arrays unless `--no-metadata` is given.

### `mizukage calib`

Inspect a `lightcal` calibration directory as text or JSON.

```bash
mizukage calib images/lightcal
mizukage calib images/lightcal --json   # full calibration data, all cameras
```

Reports geometry (intrinsics, distortion, focus bundles), colour matrices, vignetting grids, sensor characteristics, and hot-pixel statistics.

### `mizukage calib-view`

Interactive GUI explorer for a `lightcal` directory (requires `mizukage[explorer]`).

```bash
pip install 'mizukage[explorer]'
mizukage calib-view images/lightcal
```

Select a camera from the sidebar to populate six tabs:

| Tab | Contents |
|-----|----------|
| **Hot pixels** | Sqrt-scaled defect-density heatmap; per-measurement gain, temperature, exposure |
| **Noise model** | VST sigma-vs-gain curves per channel (R / Gr / Gb / B) |
| **Vignetting** | Per-camera falloff correction grid; hall-code selector for C-array cameras |
| **Geometry** | Intrinsics per focus bundle (fx, fy, cx, cy, RMS, sensor temp); radial distortion coefficients; ideal-vs-distorted grid visualisation; rotation matrix and camera world position |
| **Color** | Factory forward and colour matrices per illuminant; neutral-point locus scatter plot (rg/bg ratios) |
| **Layout** | Bird's-eye position map for all 16 modules |

The sidebar shows sensor black/white levels, device model, and calibration date.

---

## Python API

### Opening a file

```python
import mizukage

lri = mizukage.open_lri("photo.lri")
```

### Metadata

```python
meta = lri.metadata
print(meta.focal_length_mm)   # e.g. 28.0
print(meta.device_model)      # "L16"
print(meta.gps)               # GpsData | None
print(meta.awb_gains)         # AwbGains(r, gr, gb, b) | None
```

### Accessing camera images

```python
for img in lri.images:
    print(img.camera_id, img.width, img.height)

img = lri.get_image(mizukage.CameraId.B4)
```

### Exporting

```python
# 8-bit PNG — debayer + AWB + sRGB gamma
img.to_png("B4.png")

# 16-bit TIFF, linear light, +0.5 EV
img.to_tiff("B4.tiff", gamma="linear", exposure=0.5)

# Raw Bayer as NumPy array (uint16)
bayer = img.to_bayer_numpy()

# Debayered float32 (H, W, 3) in linear light
rgb = img.to_debayered_numpy()
```

### Calibration-aware export

```python
from pathlib import Path
from mizukage._calib import load_hot_pixel_map, load_distortion_params, load_vignetting_grid
from mizukage._types import CameraId

calib_dir = Path("images/lightcal")
cam = CameraId.B4

img.to_png(
    "B4_calib.png",
    hot_pixel_map=load_hot_pixel_map(calib_dir, cam),
    vignetting_grid=load_vignetting_grid(calib_dir, cam),
    distortion_params=load_distortion_params(calib_dir, cam),
    undistort=True,
)
```

### Depth maps (LRIS)

```python
lris = mizukage.open_lris("photo.lris")
depth = lris.depth_map       # float32 NumPy array, metres
conf  = lris.confidence_map  # float32, 0-1
```

---

## File format

`.lri` and `.lris` files are **LELR block streams**: a sequence of 32-byte headers, each followed by a protobuf payload. The magic bytes `LELR` open every block.

| Block type | Content |
|------------|---------|
| `LIGHT_HEADER` (0) | Capture metadata, camera settings, colour profiles, calibration data |
| `VIEW_PREFERENCES` (1) | App-level display preferences |
| `GPS_DATA` (2) | GPS coordinates and timestamp |

Image data (raw Bayer or JPEG-compressed Bayer) is stored as binary blobs with offsets recorded in the `LIGHT_HEADER` protobuf. `mizukage` reads these via direct byte slices without copying the entire payload into protobuf fields.

Calibration files use the same LELR format. `calibration.lri` holds per-camera `FactoryModuleCalibration` blocks; `hotpixel.rec` embeds zlib-compressed defect bitmaps.

---

## Camera layout

| Array | Cameras | Focal length |
|-------|---------|--------------|
| A | A1-A5 | 28 mm eq. |
| B | B1-B5 | 70 mm eq. |
| C | C1-C6 | 150 mm eq. |

C-array cameras use a movable mirror to extend effective aperture. Their calibration includes per-hall-code vignetting grids and mirror actuator mapping.

---

## Requirements

- Python 3.12+
- NumPy >= 1.26, SciPy >= 1.11, Pillow >= 10, protobuf >= 5, click >= 8.1, rich >= 13
