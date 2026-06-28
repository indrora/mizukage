# Pixel Encoding Formats

The L16 camera stores raw sensor data using one of two encoding schemes,
identified by the `Surface.format` field in the protobuf:

| `format` value | `RawFormat` | Description |
|---|---|---|
| 0 | `BAYER_JPEG` | Four 8-bit JPEGs summed to reconstruct 10-bit data |
| 7 | `PACKED_10BPP` | 5 bytes → 4 pixels at 10 bits each |
| 8 | `PACKED_12BPP` | 12-bit packed (rare; same unpacking logic, different geometry) |
| 9 | `PACKED_14BPP` | 14-bit packed (rare) |

Both formats produce a uint16 array of shape `(height, width)` after
unpacking.

---

## PACKED_10BPP

The most common format. Stores four 10-bit pixel values in five consecutive
bytes using little-endian bit packing.

### Bit layout

```
Byte 0  Byte 1  Byte 2  Byte 3  Byte 4
[p0_7:0][p1_1:0|p0_9:8][p2_3:0|p1_9:2][p3_5:0|p2_9:4][p3_9:6]
```

Decoding four pixels from bytes b0–b4:

```
p0 = b0        |  ((b1 & 0x03) << 8)
p1 = (b1 >> 2) |  ((b2 & 0x0F) << 6)
p2 = (b2 >> 4) |  ((b3 & 0x3F) << 4)
p3 = (b3 >> 6) |   (b4         << 2)
```

### Row stride

The `Surface.row_stride` field gives the byte width of one row of packed
data. This may be wider than the minimum needed for `width` pixels (e.g.
for alignment). The unpacker uses `row_stride` directly:

```python
raw = np.frombuffer(data, np.uint8, offset=abs_offset,
                    count=row_stride * height).reshape(height, row_stride)
# Process only the first (width // 4) * 5 bytes per row
groups_per_row = width // 4
g = raw[:, :groups_per_row * 5].reshape(height, groups_per_row, 5).astype(np.uint16)
```

### Value range

After unpacking: `[0 .. 1023]` (10-bit, unsigned).

After black-level subtraction (default 64): `[0 .. 959]`.

---

## BAYER_JPEG

An unusual encoding used in some captures (the exact conditions that trigger
it are not documented). Instead of storing raw packed bits, the camera
decomposes the 10-bit Bayer image into four 8-bit JPEG files which are
summed to recover the 10-bit precision.

### How the 10-bit reconstruction works

A 10-bit pixel value `v` is written as the sum of four 8-bit JPEG pixels,
each approximately `v/4`:

```
pixel ≈ jpeg0 + jpeg1 + jpeg2 + jpeg3
```

Because each JPEG is 8-bit (0–255) and carries approximately one quarter of
the signal, summing four recovers the range 0–1020 ≈ 0–1023. In practice
the library decodes each JPEG, multiplies by 4, sums, and clips:

```python
decoded = (np.array(PIL.Image.open(jpeg_bytes), np.uint32) * 4).clip(0, 1023).astype(np.uint16)
```

### BJPG header

The BAYER_JPEG data begins at `abs_offset` (= `block_start + surface.data_offset`)
with a 1576-byte header:

```
Offset  Size   Type     Field
------  -----  -------  ---------------------------
0       4      bytes    magic = b"BJPG"
4       4      u32 LE   format: 0=colour, 1=mono
8       4      u32 LE   jpeg_len[0]
12      4      u32 LE   jpeg_len[1]
16      4      u32 LE   jpeg_len[2]
20      4      u32 LE   jpeg_len[3]
24      1552   bytes    padding (unknown)
```

Total header: **1576 bytes**.

### Format 0 — Colour (four JPEGs)

Immediately after the header, four JPEG payloads follow in order,
each `jpeg_len[i]` bytes long:

```
[ header 1576 bytes ][ JPEG 0 ][ JPEG 1 ][ JPEG 2 ][ JPEG 3 ]
```

Each JPEG is a **half-resolution grayscale** image (`width/2 × height/2`).
They correspond to the four Bayer channel positions in scan order:

| JPEG index | Bayer channel |
|---|---|
| 0 | R  (at r_row, r_col) |
| 1 | G1 (at r_row, b_col) |
| 2 | G2 (at b_row, r_col) |
| 3 | B  (at b_row, b_col) |

Where `b_row = 1 - r_row`, `b_col = 1 - r_col`.

Reconstruction into a full Bayer array:

```python
out = np.zeros((height, width), np.uint16)
b_row, b_col = 1 - r_row, 1 - r_col
out[r_row::2, r_col::2] = R       # channel 0
out[r_row::2, b_col::2] = G1      # channel 1
out[b_row::2, r_col::2] = G2      # channel 2
out[b_row::2, b_col::2] = B       # channel 3
```

### Format 1 — Mono (one JPEG)

Only `jpeg_len[0]` is used. The single JPEG is a full-resolution
(`width × height`) grayscale image. Decode as above (×4, clip), return
directly without interleaving.

### `row_stride` for BAYER_JPEG

`Surface.row_stride` is 0 for BAYER_JPEG. Do not use it to size the
buffer; use the JPEG lengths from the BJPG header instead.

---

## Bayer pattern orientation

The Bayer CFA pattern (which colour sits in which quadrant of the 2×2 tile)
is given by `sensor_bayer_red_override` in the `CameraModule` proto.

The library encodes it as `(r_col | (r_row << 1))` matching
`BayerPattern`:

| r_row | r_col | Name | Value |
|---|---|---|---|
| 0 | 0 | RGGB | 0 |
| 0 | 1 | GRBG | 1 |
| 1 | 0 | GBRG | 2 |
| 1 | 1 | BGGR | 3 ← most common on L16 |

When `sensor_bayer_red_override` is absent, the library defaults to BGGR.
When it is present with negative sentinel values (the C6 mono sensor),
`r_row` and `r_col` are set to `None`.

---

## Black level

The sensor black level (pedestal) is stored in the `SensorCharacterization`
protobuf message. The library defaults to `64` if not found.

After subtraction, the usable pixel range for 10-bit data is:
`[0 .. 1023 - black_level]` ≈ `[0 .. 959]`.

The library clips the result to `[0, white_level]` and returns `uint16`.
