# Light L16 Camera Hardware

## Overview

The Light L16 is a computational camera containing **16 camera modules**
arranged in a flat array behind a fixed front panel. Each module is an
independent image sensor with its own lens, voice-coil motor (VCM) focus
actuator, and in some cases a mirror or prism to fold the optical path.

A single capture fires **10–11 modules simultaneously** (depending on the
selected focal length and lighting conditions). The firmware selects which
modules to activate.

---

## Module arrays

The 16 modules are divided into three focal-length groups, labelled A, B, and C:

| Array | Nominal focal length | Module count | Sensor | CFA |
|---|---|---|---|---|
| A | 28 mm | 5 (A1–A5) | AR1335 | BGGR |
| B | 70 mm | 5 (B1–B5) | AR1335 | BGGR |
| C (1–5) | 150 mm | 5 (C1–C5) | AR1335 | BGGR |
| C6 | 150 mm | 1 | AR1335_MONO | — (no Bayer) |

The 150 mm lenses use a folded optical path (periscope design) via a
mirror/prism, which is why `ModuleCapture.mirror_position` is set for
C-array modules and `None` for A/B.

---

## Module naming and integer indices

In the protobuf, each module is identified by an integer `id` from 0–15:

```
A1=0  A2=1  A3=2  A4=3  A5=4
B1=5  B2=6  B3=7  B4=8  B5=9
C1=10 C2=11 C3=12 C4=13 C5=14 C6=15
```

`CameraId.from_name("B4")` returns `CameraId.B4` (value 8). Case is ignored.

---

## Reference camera

The firmware designates one module as the **reference camera** — the one
whose framing, focal length, and exposure the user interface treats as the
primary image. This is recorded in the `LightHeader` protobuf and surfaced
as `CaptureMetadata.reference_camera`.

For 150 mm captures the reference is typically B4 or a C-array module;
for 28 mm captures it is typically an A-array module.

---

## Sensor models

| `SensorModel` | Chip | Used in | Mono? |
|---|---|---|---|
| `AR1335` | OnSemi AR1335 | A1–A5, B1–B5, C1–C5 | No |
| `AR1335_MONO` | OnSemi AR1335 (mono) | C6 | Yes |
| `IMX386` | Sony IMX386 | Later hardware variants | No |
| `IMX386_MONO` | Sony IMX386 (mono) | Later hardware variants | Yes |
| `AR835` | OnSemi AR835 | Pre-production / early units | No |

The sensor model is stored in `HwInfo` in the protobuf, not per-frame. The
library back-fills `RawImage.sensor_model` after parsing all LELR blocks.

---

## Sensor resolution

The AR1335 produces a **4160 × 3120** pixel raw image (12.99 MP) at full
resolution. Some captures may use a binned or cropped mode producing a
different resolution — always use `RawImage.width` and `RawImage.height`.

---

## Per-capture module settings

Each fired module's capture settings are in `CaptureMetadata.modules`
(a dict of `CameraId → ModuleCapture`):

```python
meta = lri.metadata
for cam_id, mc in meta.modules.items():
    print(f"{cam_id.name}: {mc.exposure_ns/1e6:.2f}ms × {mc.analog_gain}x")
```

Key per-module values:

| Field | Description |
|---|---|
| `lens_position` | VCM step count (higher = closer focus) |
| `mirror_position` | Mirror/prism step count (C-array only; `None` for A/B) |
| `analog_gain` | Sensor analog gain (1.0 = ISO 100-equivalent baseline) |
| `exposure_ns` | Exposure duration in nanoseconds |
| `digital_gain` | ISP digital gain (optional) |
| `flip_h`, `flip_v` | Sensor readout flip flags |

---

## Typical capture mode examples

**150 mm capture** (long focal length):

```
Active modules: B1–B5 (70 mm), C1–C6 (150 mm) — ~11 modules
Reference:      B4 (70 mm fallback) or a C module
Note:           C6 fires with no Bayer filter (mono), used for sharpness reference
```

**28 mm capture** (wide angle):

```
Active modules: A1–A5 (28 mm), B1–B5 (70 mm) — ~10 modules
Reference:      An A-array module
```

The camera always fires modules from at least two focal-length arrays to
enable stereo depth estimation.

---

## AWB and color calibration

Each camera module has independent factory color calibration stored as
`FactoryModuleCalibration` (field 13 in `LightHeader`). Three illuminant
modes are calibrated per module:

| CCM mode | Standard illuminant | Color temperature |
|---|---|---|
| 0 | Illuminant A (Tungsten) | ~2856 K |
| 2 | Illuminant D65 (Daylight) | 6504 K |
| 6 | Illuminant F11 / D50 (Fluorescent) | ~4000–5000 K |

Global AWB gains (applied identically across all modules) are stored in
`ViewPreferences` and surfaced as `CaptureMetadata.awb_gains`.

See [api-reference.md](api-reference.md#colorprofile) for the `ColorProfile`
data type, and [format-lri.md](format-lri.md) for the protobuf field layout.
