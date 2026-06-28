# API Reference

## Module `shadow`

### Convenience functions

```python
shadow.open_lri(path: str | Path) -> LriFile
shadow.open_lris(path: str | Path) -> LrisFile
```

Thin wrappers around `LriFile.open()` and `LrisFile.open()`.

---

## `LriFile`

Reader for a Light L16 LRI file. The entire file is read into memory once;
image data is accessed via zero-copy views.

### Class method

```python
LriFile.open(path: str | Path) -> LriFile
```

Parse all LELR blocks in the file. Raises `ValueError` for invalid files;
silently skips individual corrupt blocks.

### Properties

| Property | Type | Description |
|---|---|---|
| `images` | `list[RawImage]` | All active camera images, sorted by `CameraId`. |
| `metadata` | `CaptureMetadata` | Capture-level metadata (focal length, AWB, GPS…). |
| `color_profiles` | `dict[CameraId, list[ColorProfile]]` | Factory color calibration per camera. |
| `reference_image` | `RawImage \| None` | Image from the firmware-designated reference camera. |

### Methods

```python
image_for_camera(camera: CameraId | str | int) -> RawImage | None
```
Returns the `RawImage` for the given camera, or `None` if it wasn't fired.
Accepts `CameraId`, a name string (`"B4"`, `"c1"` — case-insensitive), or
an integer index (0–15).

```python
images_by_camera() -> dict[CameraId, RawImage]
```
Returns a mapping of `CameraId` to `RawImage` for convenient lookup.

---

## `LrisFile`

Reader for a Light L16 LRIS sidecar file (written by the Lumen desktop app).

### Class method

```python
LrisFile.open(path: str | Path) -> LrisFile
```

Raises `ValueError` if the magic number is wrong or the file is truncated.

### Properties

| Property | Type | Description |
|---|---|---|
| `depth_map` | `np.ndarray` (int32, shape 195×260) | Quantized disparity. Negative = invalid. |
| `depth_shape` | `tuple[int, int]` | Always `(195, 260)`. |
| `valid_fraction` | `float` | Fraction of pixels with non-negative disparity. |
| `disparity_range` | `tuple[int, int]` | `(min, max)` disparity among valid pixels; `(0,0)` if none. |

---

## `RawImage`

Raw sensor data for one camera module. Pixel data is decoded lazily on demand
from the source file bytes.

### Public fields (set by `LriFile`)

| Field | Type | Description |
|---|---|---|
| `camera_id` | `CameraId` | Which camera this is (A1–C6). |
| `sensor_model` | `SensorModel` | Sensor chip model. |
| `width` | `int` | Sensor width in pixels. |
| `height` | `int` | Sensor height in pixels. |
| `raw_format` | `RawFormat` | Pixel encoding: `PACKED_10BPP` or `BAYER_JPEG`. |
| `bayer_r_row` | `int \| None` | Row of R pixel in 2×2 Bayer tile (0 or 1). `None` for mono. |
| `bayer_r_col` | `int \| None` | Column of R pixel in 2×2 Bayer tile. `None` for mono. |
| `analog_gain` | `float` | Sensor analog gain at capture. |
| `exposure_ns` | `int` | Exposure duration in nanoseconds. |
| `digital_gain` | `float \| None` | Sensor digital gain, if recorded. |
| `flip_h` | `bool` | Sensor horizontal flip. |
| `flip_v` | `bool` | Sensor vertical flip. |
| `color_profiles` | `list[ColorProfile]` | Factory color calibration (back-filled from all blocks). |

### Properties

```python
cfa_pattern -> BayerPattern | None
```
The Bayer CFA layout (`RGGB`, `GRBG`, `GBRG`, or `BGGR`). `None` for mono
sensors. **Important**: `BayerPattern.RGGB == 0` is falsy — always use
`if img.cfa_pattern is not None`, never bare truthiness.

```python
is_mono -> bool
```
`True` if the sensor has no Bayer filter (e.g. AR1335_MONO on C6).

```python
exposure_ms -> float
```
`exposure_ns / 1_000_000` — convenience in milliseconds.

```python
white_level -> int
```
Usable maximum after black subtraction: `1023 - black_level`.

### Methods

```python
color_profile(illuminant: Illuminant) -> ColorProfile | None
```
Returns the factory `ColorProfile` for the given illuminant, or `None`.

```python
to_raw_numpy(*, subtract_black: bool = True) -> np.ndarray
```
Decode pixel data to a uint16 array of shape `(height, width)`. No
demosaicing is applied — the returned array is the raw Bayer mosaic (or mono
channel). Values are in `[0..white_level]` by default, or `[0..1023]` with
`subtract_black=False`.

```python
to_debayered_numpy(
    *, half_res: bool = False, subtract_black: bool = True
) -> np.ndarray
```
Demosaic the Bayer array to RGB.

- `half_res=False` (default): bilinear full-resolution, returns float32
  `(height, width, 3)` with values in `[0..white_level]`.
- `half_res=True`: fast subsample, returns uint16 `(height/2, width/2, 3)`.

For mono sensors, returns the single channel replicated to three identical
channels.

```python
to_png(
    path: str | Path,
    *, raw: bool = False,
    half_res: bool = False,
    subtract_black: bool = True,
) -> None
```
Save as PNG.

- `raw=False` (default): 8-bit RGB (demosaiced). `half_res` controls resolution.
- `raw=True`: 16-bit grayscale Bayer (no demosaic, scaled from 10-bit to 16-bit).

Note: Pillow cannot write 16-bit RGB PNG. For 16-bit per-channel debayered
output use `to_tiff()` or `to_debayered_numpy()` with `tifffile`.

```python
to_tiff(
    path: str | Path,
    *, raw: bool = False,
    half_res: bool = False,
    subtract_black: bool = True,
) -> None
```
Save as TIFF. Same `raw`/`half_res` semantics as `to_png()`.

---

## Data Types

### `CaptureMetadata`

Top-level metadata for an LRI capture. Fields may be `None` if absent from
the file.

| Field | Type | Description |
|---|---|---|
| `image_id` | `tuple[int, int]` | `(unique_id_low, unique_id_high)` from the firmware. |
| `focal_length_mm` | `int \| None` | Nominal focal length of the active array (28/70/150). |
| `reference_camera` | `CameraId \| None` | Camera the firmware chose as the primary. |
| `device_model` | `str` | Device string, e.g. `"L16"`. |
| `firmware_version` | `str` | Firmware string. |
| `awb_gains` | `AwbGains \| None` | White-balance channel multipliers. |
| `awb_mode` | `AwbMode \| None` | Auto-WB mode used. |
| `hdr_mode` | `HdrMode \| None` | HDR rendering mode. |
| `scene_mode` | `SceneMode \| None` | Scene classifier result. |
| `on_tripod` | `bool \| None` | Tripod detection result. |
| `gps` | `GpsData \| None` | GPS position at capture. |
| `modules` | `dict[CameraId, ModuleCapture]` | Per-camera capture settings. |

### `AwbGains`

```python
@dataclass(slots=True)
class AwbGains:
    r: float    # red channel gain
    gr: float   # green-in-red channel gain
    gb: float   # green-in-blue channel gain
    b: float    # blue channel gain
```

### `ColorProfile`

Factory color calibration for one camera under one illuminant.

```python
@dataclass(slots=True)
class ColorProfile:
    camera_id: CameraId
    illuminant: Illuminant
    forward_matrix: tuple[float, ...]   # 9 floats, row-major (sensor RGB → XYZ)
    color_matrix: tuple[float, ...]     # 9 floats, row-major (inverse)
    rg_ratio: float                     # per-capture illuminant chromaticity
    bg_ratio: float
```

### `ModuleCapture`

Per-camera capture settings recorded at exposure time.

```python
@dataclass(slots=True)
class ModuleCapture:
    camera_id: CameraId
    enabled: bool
    lens_position: int        # VCM step count
    mirror_position: int | None   # mirror step count (if present)
    analog_gain: float
    exposure_ns: int
    digital_gain: float | None
    flip_h: bool
    flip_v: bool
```

### `GpsData`

```python
@dataclass(slots=True)
class GpsData:
    latitude: float
    longitude: float
    altitude_m: float | None
    heading: float | None
    speed: float | None
```

---

## Enumerations

### `CameraId`

Maps camera module names to integer indices (used in the protobuf).

| Name | Value | Focal array |
|---|---|---|
| A1–A5 | 0–4 | 28 mm (5 modules) |
| B1–B5 | 5–9 | 70 mm (5 modules) |
| C1–C6 | 10–15 | 150 mm (6 modules) |

Helper properties: `.array` → `"A"`, `"B"`, or `"C"`; `.index` → 1-based
position within the array.

Class method: `CameraId.from_name("b4")` → `CameraId.B4` (case-insensitive).

### `SensorModel`

| Name | Value | Notes |
|---|---|---|
| `UNKNOWN` | 0 | Sentinel before back-fill |
| `AR835` | 1 | |
| `AR1335` | 2 | Most common colour sensor |
| `AR1335_MONO` | 3 | C6 — no Bayer filter |
| `IMX386` | 4 | |
| `IMX386_MONO` | 5 | |

Property: `.is_mono` → `True` for `AR1335_MONO` and `IMX386_MONO`.

### `RawFormat`

| Name | Value | Description |
|---|---|---|
| `BAYER_JPEG` | 0 | Four 8-bit JPEGs summed to 10-bit |
| `PACKED_10BPP` | 7 | 5 bytes → 4 pixels, 10-bit |
| `PACKED_12BPP` | 8 | 12-bit packed (uncommon) |
| `PACKED_14BPP` | 9 | 14-bit packed (uncommon) |

### `BayerPattern`

Encodes the position of the R pixel in the 2×2 Bayer tile as
`(r_col | (r_row << 1))`.

| Name | Value | R position |
|---|---|---|
| `RGGB` | 0 | row 0, col 0 |
| `GRBG` | 1 | row 0, col 1 |
| `GBRG` | 2 | row 1, col 0 |
| `BGGR` | 3 | row 1, col 1 ← most common on L16 |

Properties: `.r_row`, `.r_col`.

**Gotcha**: `BayerPattern.RGGB == 0` is falsy. Use `is not None` checks.

### `Illuminant`

| Name | Value | Description |
|---|---|---|
| `A` | 0 | Tungsten (~2856 K) |
| `D50` | 1 | Horizon daylight (~5003 K) |
| `D65` | 2 | Standard daylight (6504 K) |
| `D75` | 3 | North sky daylight (~7504 K) |
| `F2` | 4 | Cool white fluorescent |
| `F7` | 5 | Broadband fluorescent |
| `F11` | 6 | Narrow-band fluorescent (~4000 K) |
| `TL84` | 7 | TL84 shop lamp |
| `UNKNOWN` | 99 | |

### `HdrMode`

`NONE`, `DEFAULT`, `NATURAL`, `SURREAL`

### `SceneMode`

`PORTRAIT`, `LANDSCAPE`, `SPORT`, `MACRO`, `NIGHT`, `NONE`

### `AwbMode`

`AUTO`, `DAYLIGHT`, `SHADE`, `CLOUDY`, `TUNGSTEN`, `FLUORESCENT`,
`FLASH`, `CUSTOM`, `KELVIN`

### `Orientation`

`NORMAL`, `ROT90_CW`, `ROT90_CCW`, `ROT90_CW_VFLIP`, `ROT90_CCW_VFLIP`,
`VFLIP`, `HFLIP`, `ROT180`
