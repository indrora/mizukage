# shadow documentation

`shadow` is a Python 3.12 library and CLI for reading and extracting data
from **Light L16** multi-camera LRI/LRIS files.

## Contents

### Getting started

- [quickstart.md](quickstart.md) — Installation, basic usage, first export

### Reference

- [api-reference.md](api-reference.md) — `LriFile`, `LrisFile`, `RawImage`, all types and enumerations
- [cli-reference.md](cli-reference.md) — `shadow info`, `shadow extract`, `shadow export`

### File formats

- [format-lri.md](format-lri.md) — LELR block structure, LightHeader protobuf layout, image data
- [format-lris.md](format-lris.md) — LRIS sidecar format (depth map, header)
- [format-pixels.md](format-pixels.md) — PACKED_10BPP bit packing and BAYER_JPEG encoding

### Hardware and schema

- [camera-hardware.md](camera-hardware.md) — L16 module arrays, sensor models, focal lengths, AWB
- [protobuf-schema.md](protobuf-schema.md) — All 29 protobuf messages and their fields

### Internals

- [architecture.md](architecture.md) — Module responsibilities, key invariants, proto compilation, extension guide

---

## Quick links

```python
import shadow

lri  = shadow.open_lri("photo.lri")      # LriFile
lris = shadow.open_lris("photo.lris")    # LrisFile

ref = lri.reference_image                # RawImage
ref.to_png("out.png")                    # 8-bit RGB PNG
ref.to_png("out_raw.png", raw=True)      # 16-bit grayscale Bayer
arr = ref.to_raw_numpy()                 # numpy uint16 (H, W)
```

```bash
shadow info   photo.lri
shadow export photo.lri ./out/
shadow extract photo.lri ./raw/ --camera B4
```
