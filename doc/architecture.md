# Architecture

## Design philosophy

`shadow` is **library-first**. The CLI is ~90% calls into the library; all
interesting logic lives in the `shadow` package. The public API surface is
small and Pythonic — callers work with `LriFile`, `RawImage`, and plain
dataclasses, never with raw protobuf objects.

---

## Module map

```
shadow/
  __init__.py         public re-exports; open_lri() / open_lris() helpers
  _types.py           enums and dataclasses; no I/O, no numpy
  _block.py           LELR binary header parser
  _proto.py           protobuf → _types bridge (only file that imports shadow.proto.*)
  _unpack.py          vectorised pixel decoders (10bpp, BayerJPEG)
  _debayer.py         numpy-only demosaicing (half-res and bilinear)
  _image.py           RawImage class
  _lri.py             LriFile orchestration
  _lris.py            LrisFile sidecar reader
  proto/
    __init__.py       sys.path patch for flat inter-_pb2 imports
    *_pb2.py          generated from protobuf/*.proto (committed to repo)
  cli/
    main.py           click group definition
    commands/
      info.py         `shadow info`
      extract.py      `shadow extract`
      export.py       `shadow export`
```

---

## Layer responsibilities

### `_types.py`

Zero I/O, zero numpy. Pure data — enums (`CameraId`, `RawFormat`,
`BayerPattern`, `Illuminant`, …) and dataclasses (`CaptureMetadata`,
`RawImage`-adjacent types). All other modules import from here. Nothing
else should define data structures.

### `_block.py`

Knows only the binary LELR header format. Exposes `iter_blocks(data: bytes)`
which yields `(block_start_offset, BlockHeader)` pairs. Has no knowledge of
protobuf or pixels.

### `_proto.py`

The **only** file that imports `shadow.proto.*`. All other modules see
plain Python objects from `_types.py`. Functions in this file translate
protobuf messages into those types:

- `parse_light_header(data: bytes) -> LightHeader` — raw parse
- `update_metadata_from_light_header(meta, lh)` — in-place merge
- `update_metadata_from_view_prefs(meta, vp)` — in-place merge
- `update_metadata_from_gps_block(meta, gps)` — in-place merge
- `sensor_id_map_from_hw_info(hw_info) -> dict[CameraId, SensorModel]`
- `color_profiles_from_calibrations(cals) -> dict[CameraId, list[ColorProfile]]`
- `bayer_pos_from_sbro(sbro) -> tuple[int, int] | None`
- `black_level_from_sensor_data(sensor_data_list) -> float`

Keeping protobuf imports isolated here means:
1. The rest of the library is testable without compiled proto files.
2. Upgrading the protobuf schema only requires changes in one file.

### `_unpack.py`

Vectorised pixel decoders — no loops, pure numpy operations.

`unpack_10bpp(data, abs_offset, width, height, stride) -> np.ndarray`  
Reshapes the raw bytes into `(height, groups, 5)` and extracts four
10-bit values per group via bitmasking. Returns uint16 `(height, width)`.

`decode_bjpg(data, abs_offset, width, height, r_row, r_col) -> np.ndarray`  
Reads the BJPG header, decodes each JPEG with Pillow, multiplies by 4,
clips to 1023, then interleaves the four half-res channels into a full
Bayer array.

### `_debayer.py`

Two demosaicing strategies, both numpy-only (no OpenCV):

`debayer_half(bayer, r_row, r_col) -> np.ndarray`  
Subsamples: R and B direct, G averaged. Returns uint16 `(H/2, W/2, 3)`.
Fast; good for previews.

`debayer_bilinear(bayer, r_row, r_col) -> np.ndarray`  
Full-resolution bilinear interpolation using `np.pad` and rolling windows.
Returns float32 `(H, W, 3)`. Slower; suitable for final output.

### `_image.py` — `RawImage`

Stores a reference to the source file bytes (`_file_bytes`) and a
block-relative data offset (`_data_offset`). Pixel decoding is lazy —
`to_raw_numpy()` is called only when the user asks.

The `_file_bytes` reference keeps the entire file in memory while any
`RawImage` derived from it is alive. This is intentional: it avoids
multiple file reads and gives zero-copy access to pixel data.

### `_lri.py` — `LriFile`

Orchestration: read file → iterate blocks → accumulate metadata → back-fill.

**Back-fill pattern**: `HwInfo` (sensor models) and `FactoryModuleCalibration`
(color profiles) arrive in different blocks from the per-module image data.
`LriFile._parse()` accumulates `sensor_id_map` and `color_profiles_by_cam`
during the walk, then assigns them to every `RawImage` in a second pass.

**Deduplication**: within a single file, the same camera module can appear
in multiple `LightHeader` blocks. The logic is:
- PACKED_\*: first block wins (dict-insert guard).
- BAYER_JPEG: lowest absolute `data_offset` wins (= primary/shortest exposure).

### `_lris.py` — `LrisFile`

Simple: read magic, validate, slice `depth_map` from offset 0x28.
Exposes `depth_map`, `valid_fraction`, and `disparity_range`.

---

## Proto compilation

The `protobuf/` directory contains 29 `.proto` files (the L16's internal
schema). They are pre-compiled to `shadow/proto/*_pb2.py` and **committed
to the repository** so that users do not need `grpcio-tools` to install the
library.

To recompile after modifying `.proto` files:

```bash
uv run --extra dev python scripts/compile_protos.py
```

`scripts/compile_protos.py` calls `grpc_tools.protoc` and regenerates
`shadow/proto/__init__.py` with the `sys.path` insertion:

```python
import sys
from pathlib import Path
_here = str(Path(__file__).parent)
if _here not in sys.path:
    sys.path.insert(0, _here)
```

This patch is required because `protoc --python_out` generates files with
flat imports (`import camera_module_pb2`) rather than package-relative ones.
The `shadow.proto` package must be imported (or any `shadow.proto.*` module)
before any `_pb2` cross-imports can resolve.

---

## Key invariants

**`data_offset` is block-relative.**  
`surface.data_offset` in any `CameraModule` proto is relative to the start
of the enclosing LELR block. The absolute file offset is:
```python
abs_offset = block_start + int(surface.data_offset)
```
This is a common source of bugs in other implementations that assume
file-relative offsets.

**`BayerPattern.RGGB == 0` is falsy.**  
Always compare with `is not None`, never bare truthiness. A camera with
RGGB pattern would evaluate `False` in a plain `if img.cfa_pattern:` check.

**Mono sensor detection.**  
C6 (AR1335_MONO) has `sensor_bayer_red_override` present in the proto but
with negative sentinel values (typically -1). `bayer_pos_from_sbro()` returns
`None` for negatives. `_lri.py` then sets `bayer_r_row = bayer_r_col = None`
to mark the image as mono.

**Multiple LightHeader blocks per file.**  
There are typically 8–9. Never assume any single block contains all metadata.

---

## Adding support for new features

### Decoding LRIS calibration data

The LRIS file contains approximately 6.5 MB of unknown data beyond the depth
map, likely including refined per-capture calibration (see
[format-lris.md](format-lris.md)). To add support:

1. Reverse-engineer the structure and add a parse method in `_lris.py`.
2. Expose the result via a new `LrisFile` property.
3. No changes needed to `_types.py` unless new data types are required.

### Supporting new pixel formats

Add the `RawFormat` enum value to `_types.py`, add an unpacker function in
`_unpack.py`, and add a `case` branch in `RawImage.to_raw_numpy()`.

### New CLI commands

1. Create `shadow/cli/commands/<name>.py` with a `@click.command()`.
2. Import it in `shadow/cli/main.py` and call `cli.add_command(...)`.
