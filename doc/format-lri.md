# LRI File Format

The **Light Raw Image** (LRI) file format is the native capture format
produced by the Light L16 camera. A typical LRI is roughly 170 MB and
contains raw sensor data from 10–11 of the 16 camera modules, together
with all capture metadata encoded as Protocol Buffers.

---

## Top-level structure

An LRI file is a linear sequence of variable-length **LELR blocks**:

```
[ LELR block 0 ][ LELR block 1 ][ LELR block 2 ] ...
```

Block types present in a typical file:

| Type | msg_type byte | Count | Contents |
|---|---|---|---|
| `LIGHT_HEADER` | 0 | 8–9 | Per-module image data + all metadata |
| `VIEW_PREFERENCES` | 1 | 1 | AWB, HDR, scene, orientation settings |
| `GPS_DATA` | 2 | 0–1 | Optional GPS fix at capture time |

The file ends when either the next 4 bytes are not `LELR`, or the block
length field is zero.

---

## LELR block header

Every block starts with a 32-byte header:

```
Offset  Size  Type        Field
------  ----  ----------  ----------------
0       4     bytes       magic = b"LELR"
4       8     u64 LE      block_length    — total block size in bytes (including this header)
12      8     u64 LE      msg_offset      — offset of the protobuf payload from block start
20      4     u32 LE      msg_len         — length of the protobuf payload
24      1     u8          msg_type        — 0=LIGHT_HEADER, 1=VIEW_PREFERENCES, 2=GPS_DATA
25      7     padding     (zeroes)
```

Struct format (Python): `<4sQQIB7x` — 32 bytes total.

The **protobuf payload** sits at `block_start + msg_offset` and is
`msg_len` bytes long. Everything between the end of the payload and
`block_start + block_length` is raw pixel data.

---

## `LIGHT_HEADER` block (msg_type = 0)

The payload is a serialised `LightHeader` protobuf (see
[protobuf-schema.md](protobuf-schema.md)). Key top-level fields:

| Field | Number | Type | Description |
|---|---|---|---|
| `modules` | 12 | repeated `CameraModule` | Per-camera image data and settings |
| `module_calibration` | 13 | repeated `FactoryModuleCalibration` | Factory CCM/WB calibration |
| `sensor_data` | 16 | repeated `SensorData` | Black level and sensor characterization |
| `hw_info` | 18 | `HwInfo` | Sensor model map and hardware identifiers |
| `view_preferences` | 19 | `ViewPreferences` | Embedded AWB/HDR/scene settings |
| `gps` | 25 | `GPSData` | Embedded GPS (may also appear as a top-level GPS_DATA block) |

There are typically **8–9 LIGHT_HEADER blocks** per file. Metadata fields
such as `hw_info` and `module_calibration` appear in only some of them;
the library accumulates these across all blocks and back-fills them into
images after processing all blocks.

### `CameraModule` proto (field 12)

Carries the image data surface for one camera module.

Key fields (selected; all 0-indexed by protobuf field number):

| Field | Type | Description |
|---|---|---|
| `id` | int32 | Camera index (0=A1 … 15=C6) |
| `sensor_data_surface` | `Surface` | Where the pixel data lives |
| `sensor_analog_gain` | float | Analog gain multiplier |
| `sensor_exposure` | int64 | Exposure in nanoseconds |
| `sensor_digital_gain` | float | Digital gain (optional) |
| `sensor_is_horizontal_flip` | bool | Horizontal readout flip |
| `sensor_is_vertical_flip` | bool | Vertical readout flip |
| `sensor_bayer_red_override` | `Point2I` | R pixel position within 2×2 tile |

#### `Surface` proto

| Field | Type | Description |
|---|---|---|
| `format` | int32 | Pixel format code (0=BAYER_JPEG, 7=PACKED_10BPP, 8=12BPP, 9=14BPP) |
| `data_offset` | int64 | **Block-relative** byte offset to the pixel data |
| `size` | `Point2I` | `x` = width, `y` = height |
| `row_stride` | int32 | Row stride in bytes (0 for BAYER_JPEG) |

**Critical**: `data_offset` is relative to the **start of the enclosing
LELR block** (not the file start). To get the absolute file offset:

```python
abs_offset = block_start + surface.data_offset
```

#### Bayer pattern (`sensor_bayer_red_override`)

A `Point2I` with `.x` = column of R pixel, `.y` = row of R pixel within the
2×2 Bayer tile. Negative sentinel values (`-1`) indicate a mono sensor with
no Bayer filter.

Decoding:

```python
r_col = sbro.x % 2   # 0 or 1
r_row = sbro.y % 2   # 0 or 1
# BayerPattern = (r_col | (r_row << 1))
```

Default when the field is absent: `BGGR` (r_row=1, r_col=1), which is the
most common pattern on L16 AR1335 sensors.

### `FactoryModuleCalibration` proto (field 13)

Per-camera factory color calibration. Contains repeated `ColorCalibration`
entries (field 2), one per illuminant mode.

Each `ColorCalibration` entry:

| Field | Type | Description |
|---|---|---|
| `f1` | int32 | Illuminant mode: 0=Tungsten(A), 2=Daylight(D65), 6=Fluorescent(F11/D50) |
| `f2` | `Matrix3x3F` | Forward CCM (sensor RGB → XYZ), fields `x00`…`x22` |
| `f3` | `Matrix3x3F` | Inverse CCM |
| `f4` | float | Illuminant chromaticity parameter (proprietary, not CIE xy) |
| `f5` | float | Illuminant chromaticity parameter |

See [CCM_ILLUMINANTS.md in known-documentation](../known-documentation/l16-pipeline-main/docs/reverse_engineering/12_CCM_ILLUMINANTS.md)
for the illuminant mode → standard illuminant mapping.

### `SensorData` and black level (field 16)

The `SensorData` message (repeated) contains `SensorCharacterization` which
carries the sensor's black level. Default: 64.0. After black subtraction,
10-bit values range from 0 to `1023 - black_level` ≈ 959.

### `HwInfo` proto (field 18)

Maps camera indices to sensor models. The library uses this to populate
`RawImage.sensor_model` via a back-fill pass after reading all blocks.

---

## Image data layout within a block

```
[ 32-byte LELR header ]
[ protobuf payload (msg_len bytes) ]
[ gap / alignment bytes ]
[ raw pixel data for one or more modules ]
```

The `Surface.data_offset` jumps directly to the start of a module's pixel
data within the block. Multiple modules' data can coexist in one block.

---

## Deduplication

Because the same camera may appear in multiple LIGHT_HEADER blocks (e.g.
different exposure brackets in HDR or duplicated metadata blocks):

- **PACKED_\*** formats: the **first** occurrence per camera wins.
- **BAYER_JPEG** format: keep the occurrence with the **lowest** absolute
  `data_offset` (this corresponds to the shortest/primary exposure).

---

## Multi-block metadata accumulation

The following are spread across blocks and must be collected in a single pass:

| Data | Source | Block pattern |
|---|---|---|
| Sensor models | `HwInfo` (field 18) | Usually one block |
| Color profiles | `FactoryModuleCalibration` (field 13) | Usually one block |
| AWB/scene/HDR | `ViewPreferences` (field 19 embedded, or standalone block type 1) | Any |
| GPS | `GPSData` (field 25 embedded, or standalone block type 2) | Optional |

The library performs a back-fill: after iterating all blocks, each `RawImage`
is assigned its sensor model and color profiles from the accumulated maps.
