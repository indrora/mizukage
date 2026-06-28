# LRIS File Format

An **LRIS** (Light Raw Image Sidecar) file is produced by the **Lumen**
desktop application, not by the camera itself. It is typically placed in
the same directory as the originating LRI file with the same base name and
a `.lris` extension.

LRIS files contain a quantized disparity/depth map computed by Lumen's
stereo reconstruction pipeline, plus additional calibration data generated
offline.

---

## Header

```
Offset  Size  Type     Field
------  ----  -------  -----------------
0x00    4     u32 LE   magic = 0x12345678
0x04    4     u32 LE   (unknown — file version or flags)
0x08    4     u32 LE   (unknown)
0x0C    4     u32 LE   (unknown)
0x10    4     u32 LE   (unknown)
0x14    4     u32 LE   (unknown)
0x18    4     u32 LE   (unknown)
0x1C    4     u32 LE   (unknown)
```

Total header before depth map: **40 bytes** (0x28).

---

## Depth map (`0x28` – `0x31827`)

A flat array of **`int32` little-endian** values beginning at file offset
`0x28` (40 bytes):

```
Shape:  195 rows × 260 columns
Size:   195 × 260 × 4 = 202,800 bytes
Dtype:  int32 LE (<i4)
```

### Value encoding

- **Negative values** indicate invalid or occluded pixels (the stereo
  matcher could not find a correspondence). The specific negative values
  are proprietary sentinel codes.
- **Non-negative values** are quantized disparity. Higher values correspond
  to objects closer to the camera. The exact disparity-to-depth conversion
  requires the camera's baseline distance and focal length, which are not
  stored in the LRIS file itself.

In practice, fresh LRIS files from unprocessed shots often show 0% valid
pixels (all-negative) because Lumen may not have finished computing depth.

### Example

```python
import shadow, numpy as np

lris = shadow.open_lris("photo.lris")
dm = lris.depth_map              # int32 (195, 260)

valid = dm[dm >= 0]
print(f"Valid: {lris.valid_fraction:.1%}")
print(f"Disparity range: {lris.disparity_range}")

# Visualise as an 8-bit depth map
if valid.size > 0:
    lo, hi = lris.disparity_range
    normed = np.clip((dm.astype(np.float32) - lo) / (hi - lo), 0, 1)
    normed[dm < 0] = 0           # mask invalid
    import PIL.Image
    PIL.Image.fromarray((normed * 255).astype(np.uint8)).save("depth.png")
```

---

## Remainder of file

After the depth map the file contains additional sections whose structure
is not fully reverse-engineered:

| Approximate offset | Size | Description |
|---|---|---|
| `0x31828` | ~6 bytes | Unknown (possibly alignment/padding) |
| `0x3182E` | ~6.5 MB | Unknown blob (possibly refined calibration or disparity confidence map) |
| Near EOF | variable | Protobuf payload — appears to contain per-capture calibration data |

The `shadow` library currently exposes only the depth map from LRIS files.
Contributions to decode the remaining sections are welcome; see
[architecture.md](architecture.md).

---

## File size

A typical LRIS file is approximately **9 MB**. The file ends with an unknown
trailer; the library does not parse anything beyond the depth map array.
