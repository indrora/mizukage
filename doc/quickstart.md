# Quickstart

## Installation

`shadow` requires Python 3.12 or newer. Install with [uv](https://github.com/astral-sh/uv) (recommended) or pip:

```bash
# With uv
uv add shadow

# Or directly into a virtual environment
pip install shadow
```

Dependencies pulled in automatically: `protobuf>=5.0`, `Pillow>=10.0`,
`numpy>=1.26`, `click>=8.1`, `rich>=13.0`.

---

## Opening an LRI file

```python
import shadow

lri = shadow.open_lri("L16_00009.lri")

# Top-level capture metadata
print(lri.metadata.focal_length_mm)       # e.g. 150
print(lri.metadata.reference_camera)      # e.g. CameraId.B4
print(lri.metadata.firmware_version)      # e.g. "2.1.0-1234"

# All active camera images (sorted by CameraId)
print(len(lri.images))                    # typically 10–11

# Look up a specific camera
img = lri.image_for_camera("B4")
print(img.width, img.height)             # e.g. 4160 3120
print(img.cfa_pattern)                   # BayerPattern.BGGR
print(img.exposure_ms)                   # e.g. 4.17
```

---

## Exporting images

```python
# Get the camera the firmware designated as the primary
ref = lri.reference_image

# 8-bit RGB PNG (bilinear demosaiced)
ref.to_png("reference.png")

# 16-bit grayscale PNG — raw Bayer data, no demosaic
ref.to_png("reference_raw.png", raw=True)

# Fast half-resolution preview
ref.to_png("preview.png", half_res=True)

# TIFF instead of PNG
ref.to_tiff("reference.tiff")

# Export every camera in a loop
import pathlib
out = pathlib.Path("./exported")
out.mkdir(exist_ok=True)

for img in lri.images:
    img.to_png(out / f"{img.camera_id.name}.png")
```

---

## Working with numpy arrays

```python
# Raw uint16 Bayer array, shape (height, width)
arr = ref.to_raw_numpy()
print(arr.shape, arr.dtype)   # (3120, 4160) uint16
print(arr.min(), arr.max())   # 0 .. ~959 after black subtraction

# Without black-level subtraction (raw 10-bit values, 0..1023)
arr_raw = ref.to_raw_numpy(subtract_black=False)

# Full-resolution RGB, shape (height, width, 3), float32 in [0..~959]
rgb = ref.to_debayered_numpy()

# Half-resolution RGB, shape (height/2, width/2, 3), uint16
rgb_half = ref.to_debayered_numpy(half_res=True)
```

---

## Opening an LRIS sidecar

LRIS files are written by the Lumen desktop app alongside an LRI. They
contain a low-resolution disparity/depth map.

```python
lris = shadow.open_lris("L16_00009.lris")

dm = lris.depth_map           # numpy int32, shape (195, 260)
print(dm.dtype, dm.shape)    # int32 (195, 260)

# Negative values = invalid/occluded pixels
valid = dm[dm >= 0]
print(lris.valid_fraction)   # e.g. 0.72 (72% valid)
print(lris.disparity_range)  # (min_disp, max_disp) among valid pixels
```

---

## CLI quickstart

```bash
# Metadata overview
shadow info photo.lri

# Metadata in JSON
shadow info photo.lri --json

# Show raw LELR block structure
shadow info photo.lri --blocks

# Export all cameras as PNG (8-bit RGB, demosaiced)
shadow export photo.lri ./out/

# Export a single camera as raw 16-bit grayscale TIFF
shadow export photo.lri ./out/ --camera B4 --raw --format tiff

# Extract raw Bayer numpy arrays
shadow extract photo.lri ./raw/ --camera B4 --camera C1
```

See [cli-reference.md](cli-reference.md) for the full flag reference.
