# CLI Reference

The `shadow` CLI is installed as a single entry-point. All sub-commands
accept `--help`.

```
shadow [OPTIONS] COMMAND [ARGS]...
```

Global options:

| Option | Description |
|---|---|
| `--version` | Print the installed version and exit. |
| `--help` | Show help and exit. |

---

## `shadow info`

Display metadata for an LRI or LRIS file.

```
shadow info [OPTIONS] FILE
```

Auto-detects `.lris` files by extension; everything else is treated as an LRI.

### Arguments

| Argument | Description |
|---|---|
| `FILE` | Path to the `.lri` or `.lris` file. |

### Options

| Option | Description |
|---|---|
| `--blocks` | Show a table of raw LELR block structure (offset, type, sizes). |
| `--cameras` | Show per-camera capture settings (always shown unless `--blocks` is the only flag). |
| `--json` | Output as JSON instead of Rich formatted tables. |
| `--help` | Show help. |

### LRI output (default)

Without flags, prints:

- A summary panel: file size, image count, focal length, reference camera,
  device model, firmware version, AWB mode/gains, tripod flag, GPS.
- A camera table: one row per fired camera with sensor model, format (10BPP /
  JPEG / etc.), resolution, CFA pattern, exposure, analog gain, and flip flags.
  The reference camera is marked with `*`.

### LRI output (`--blocks`)

Adds a second table showing each LELR block: index, file offset (hex), block
type (LIGHT_HEADER / VIEW_PREFERENCES / GPS_DATA), total block size, and
protobuf payload offset/size within the block.

### LRIS output

Shows: file size, depth map shape (195×260), fraction of valid disparity
pixels, and disparity range.

### JSON schema (LRI)

```json
{
  "file": "photo.lri",
  "size_bytes": 178237440,
  "image_count": 11,
  "focal_length_mm": 150,
  "reference_camera": "B4",
  "device_model": "L16",
  "firmware_version": "2.1.0-1234",
  "hdr_mode": "NONE",
  "scene_mode": "LANDSCAPE",
  "awb_mode": "AUTO",
  "awb_gains": {"r": 1.82, "gr": 1.0, "gb": 1.0, "b": 1.51},
  "on_tripod": false,
  "gps": {"latitude": 37.77, "longitude": -122.41, "altitude_m": 42.0,
          "heading": null, "speed": null},
  "cameras": [
    {
      "camera_id": "B1",
      "sensor": "AR1335",
      "width": 4160,
      "height": 3120,
      "format": "PACKED_10BPP",
      "cfa": "BGGR",
      "exposure_ms": 4.17,
      "analog_gain": 1.0
    }
  ]
}
```

### JSON schema (LRIS)

```json
{
  "file": "photo.lris",
  "size_bytes": 9437184,
  "depth_shape": [195, 260],
  "valid_fraction": 0.72,
  "disparity_range": [12, 4096]
}
```

### Examples

```bash
shadow info photo.lri
shadow info photo.lri --json | jq '.cameras[].exposure_ms'
shadow info photo.lri --blocks
shadow info photo.lris
```

---

## `shadow extract`

Extract raw Bayer pixel data as NumPy `.npy` files.

```
shadow extract [OPTIONS] FILE [OUT_DIR]
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `FILE` | — | Path to the `.lri` file. |
| `OUT_DIR` | `.` | Directory to write output files. Created if missing. |

### Options

| Option | Description |
|---|---|
| `-c`, `--camera CAMERA` | Camera to extract (e.g. `A1`, `B3`). Repeatable. Omit to extract all. |
| `--no-subtract-black` | Skip black level subtraction (values will be in raw `[0..1023]` range). |
| `--no-metadata` | Skip writing `metadata.json`. |
| `--help` | Show help. |

### Output files

| File | Description |
|---|---|
| `{CAMERA}.npy` | uint16 numpy array, shape `(height, width)`. |
| `metadata.json` | Capture metadata and per-camera settings (see below). |

The `.npy` files can be loaded with:

```python
import numpy as np
arr = np.load("B4.npy")   # shape (3120, 4160), dtype uint16
```

### `metadata.json` schema

```json
{
  "focal_length_mm": 150,
  "reference_camera": "B4",
  "device_model": "L16",
  "firmware_version": "2.1.0-1234",
  "hdr_mode": "NONE",
  "scene_mode": "LANDSCAPE",
  "awb_mode": "AUTO",
  "awb_gains": {"r": 1.82, "gr": 1.0, "gb": 1.0, "b": 1.51},
  "on_tripod": false,
  "gps": null,
  "cameras": {
    "B4": {
      "sensor": "AR1335",
      "width": 4160,
      "height": 3120,
      "format": "PACKED_10BPP",
      "cfa": "BGGR",
      "exposure_ms": 4.17,
      "analog_gain": 1.0,
      "digital_gain": null,
      "flip_h": false,
      "flip_v": false,
      "black_level": 64.0
    }
  }
}
```

### Examples

```bash
# Extract all cameras to current directory
shadow extract photo.lri

# Extract two cameras to a specific directory
shadow extract photo.lri ./raw/ --camera B4 --camera C1

# Extract raw values without black subtraction
shadow extract photo.lri ./raw/ --no-subtract-black
```

---

## `shadow export`

Export camera images as PNG or TIFF files.

```
shadow export [OPTIONS] FILE [OUT_DIR]
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `FILE` | — | Path to the `.lri` file. |
| `OUT_DIR` | `.` | Directory to write images. Created if missing. |

### Options

| Option | Default | Description |
|---|---|---|
| `-c`, `--camera CAMERA` | all | Camera to export. Repeatable. |
| `-f`, `--format {png\|tiff}` | `png` | Output image format. |
| `--raw` | off | Export 16-bit grayscale Bayer (no demosaicing). |
| `--half-res` | off | Use fast half-resolution debayer (ignored with `--raw`). |
| `--no-subtract-black` | off | Skip black level subtraction. |
| `--help` | | Show help. |

### Output files

Output filenames follow the pattern `{CAMERA}{suffix}.{ext}`:

| Mode | Suffix | Example |
|---|---|---|
| Debayered color (default) | (none) | `B4.png` |
| Raw Bayer (`--raw`) | `_raw` | `B4_raw.png` |
| TIFF (`--format tiff`) | | `B4.tiff`, `B4_raw.tiff` |

### Pixel format summary

| Flag | Format | Bit depth | Channels |
|---|---|---|---|
| (default) | 8-bit RGB PNG | 8 | 3 (debayered, bilinear) |
| `--half-res` | 8-bit RGB PNG | 8 | 3 (debayered, subsampled) |
| `--raw` | 16-bit grayscale PNG | 16 | 1 (raw Bayer mosaic) |
| `--format tiff` | 8-bit RGB TIFF | 8 | 3 |
| `--raw --format tiff` | 16-bit grayscale TIFF | 16 | 1 |

For 16-bit per-channel color output, use the library API directly:

```python
import tifffile, numpy as np
rgb = img.to_debayered_numpy()            # float32 (H, W, 3)
rgb16 = (rgb * (65535 / img.white_level)).clip(0, 65535).astype(np.uint16)
tifffile.imwrite("out.tiff", rgb16)
```

### Examples

```bash
# Export all cameras as 8-bit RGB PNG
shadow export photo.lri ./out/

# Export the reference camera as raw 16-bit TIFF
shadow export photo.lri ./out/ --camera B4 --raw --format tiff

# Half-res preview of all cameras
shadow export photo.lri ./preview/ --half-res

# Export two cameras without black subtraction
shadow export photo.lri ./out/ --camera A1 --camera B4 --no-subtract-black
```
