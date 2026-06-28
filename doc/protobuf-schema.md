# Protobuf Schema

The Light L16 firmware encodes all metadata as Protocol Buffers. The
`protobuf/` directory in this repository contains 29 `.proto` files
reverse-engineered from the camera's binary format. The corresponding
`shadow/proto/*_pb2.py` files are committed to the repository so that
users do not need `grpcio-tools` installed.

---

## Root message: `LightHeader`

`lightheader_pb2.LightHeader` is the container decoded from every
`LIGHT_HEADER` LELR block (msg_type = 0). All other messages are nested
within it or referenced by it.

### Top-level fields (selected)

| Field name | Field number | Type | Description |
|---|---|---|---|
| `image_unique_id_low` | 1 | int64 | Low 64 bits of a 128-bit capture UUID |
| `image_unique_id_high` | 2 | int64 | High 64 bits |
| `image_focal_length` | 3 | int32 | Nominal focal length in mm |
| `image_reference_camera` | 11 | int32 | Camera index of the reference module |
| `modules` | 12 | repeated `CameraModule` | Per-module image surfaces and settings |
| `module_calibration` | 13 | repeated `FactoryModuleCalibration` | Factory color calibration |
| `sensor_data` | 16 | repeated `SensorData` | Black level and sensor characterization |
| `hw_info` | 18 | `HwInfo` | Sensor model map |
| `view_preferences` | 19 | `ViewPreferences` | AWB, HDR, scene, orientation |
| `device_model_name` | 20 | string | e.g. `"L16"` |
| `device_fw_version` | 22 | string | Firmware version string |
| `gps_data` | 25 | `GPSData` | GPS fix at capture time |

---

## `CameraModule` (`camera_module_pb2`)

One entry per active camera module in `LightHeader.modules`.

| Field | Type | Description |
|---|---|---|
| `id` | int32 | Camera index 0–15 (maps to `CameraId`) |
| `is_enabled` | bool | Whether this module fired |
| `lens_position` | int32 | VCM step count |
| `mirror_position` | int32 | Mirror/prism step (C-array only; absent for A/B) |
| `sensor_analog_gain` | float | Analog gain |
| `sensor_exposure` | int64 | Exposure in nanoseconds |
| `sensor_digital_gain` | float | Digital gain (optional) |
| `sensor_is_horizontal_flip` | bool | H readout flip |
| `sensor_is_vertical_flip` | bool | V readout flip |
| `sensor_bayer_red_override` | `Point2I` | Position of R pixel in Bayer tile |
| `sensor_data_surface` | `Surface` | Where pixel data lives |

### `Surface`

| Field | Type | Description |
|---|---|---|
| `format` | int32 | 0=BAYER_JPEG, 7=10BPP, 8=12BPP, 9=14BPP |
| `data_offset` | int64 | Byte offset from **block start** to pixel data |
| `size` | `Point2I` | `x`=width, `y`=height |
| `row_stride` | int32 | Bytes per row (0 for BAYER_JPEG) |

---

## `FactoryModuleCalibration` (`color_calibration_pb2`)

Repeated field 13 in `LightHeader`. One per camera module.

| Field | Type | Description |
|---|---|---|
| `camera_id` | int32 | Camera index 0–15 |
| `color` | repeated `ColorCalibration` | One entry per illuminant (typically 3) |

### `ColorCalibration`

| Field | Type | Description |
|---|---|---|
| `type` | int32 | Illuminant mode: 0=Tungsten, 2=D65, 6=F11/D50 |
| `forward_matrix` | `Matrix3x3F` | Sensor RGB → XYZ (9 float fields `x00`…`x22`) |
| `color_matrix` | `Matrix3x3F` | Inverse CCM |
| `rg_ratio` | float | Per-capture illuminant chromaticity estimate (R/G) |
| `bg_ratio` | float | Per-capture illuminant chromaticity estimate (B/G) |

The illuminant mode mapping:
- **0** → Illuminant A (Tungsten, ~2856 K)
- **2** → Illuminant D65 (Standard Daylight, 6504 K)
- **6** → Illuminant F11 or D50 (~4000–5000 K)

---

## `HwInfo` (`hw_info_pb2`)

Contains hardware configuration, most importantly the sensor model for each
camera slot.

| Field | Type | Description |
|---|---|---|
| `camera` | repeated `CameraHwInfo` | One per module |

### `CameraHwInfo`

| Field | Type | Description |
|---|---|---|
| `id` | int32 | Camera index 0–15 |
| `sensor` | int32 | Sensor model code (maps to `SensorModel`) |

Sensor codes: 0=UNKNOWN, 1=AR835, 2=AR1335, 3=AR1335_MONO, 4=IMX386, 5=IMX386_MONO.

---

## `SensorData` (`sensor_characterization_pb2`)

Repeated field 16 in `LightHeader`. Holds per-sensor characterization.

Contains a `SensorCharacterization` sub-message (field `data`) with:

| Field | Type | Description |
|---|---|---|
| `black_level` | float | Sensor black pedestal (default 64.0) |

---

## `ViewPreferences` (`view_preferences_pb2`)

AWB, scene, HDR, and orientation settings. Can appear as:
- Embedded in `LightHeader` at field 19.
- As a standalone `VIEW_PREFERENCES` LELR block (msg_type = 1).

| Field | Type | Description |
|---|---|---|
| `awb_mode` | int32 | AWB mode code |
| `awb_gains` | `AwbGains` | Channel gains (fields `r`, `g_r`, `g_b`, `b`) |
| `hdr_mode` | int32 | HDR mode code |
| `scene_mode` | int32 | Scene classifier result |
| `is_on_tripod` | bool | Tripod detection |
| `orientation` | int32 | Device orientation at capture |

---

## `GPSData` (`gps_data_pb2`)

GPS position at capture time. Can appear as:
- Embedded in `LightHeader` at field 25.
- As a standalone `GPS_DATA` LELR block (msg_type = 2).

| Field | Type | Description |
|---|---|---|
| `latitude` | double | Decimal degrees |
| `longitude` | double | Decimal degrees |
| `altitude` | `FloatValue` | Optional; metres |
| `heading` | `FloatValue` | Optional; degrees from north |
| `speed` | double | Optional; m/s |

---

## Geometric and calibration messages

These are present in the proto schema and used internally by the Lumen
pipeline, but are not currently surfaced by `shadow`.

| Message | File | Description |
|---|---|---|
| `GeometricCalibration` | `geometric_calibration_pb2` | Lens distortion and extrinsics per module |
| `Distortion` | `distortion_pb2` | Radial/tangential distortion coefficients |
| `VignettingCharacterization` | `vignetting_characterization_pb2` | Per-channel vignetting map |
| `DeadPixelMap` | `dead_pixel_map_pb2` | Factory dead-pixel coordinates |
| `HotPixelMap` | `hot_pixel_map_pb2` | Hot-pixel coordinates |
| `FlashCalibration` | `flash_calibration_pb2` | (Unknown application; L16 has no flash) |
| `ToFCalibration` | `tof_calibration_pb2` | Time-of-flight calibration (proximity sensor) |
| `IMUData` | `imu_data_pb2` | Accelerometer/gyro data at capture |
| `ProximitySensors` | `proximity_sensors_pb2` | Proximity sensor readings |
| `DeviceTemp` | `device_temp_pb2` | Thermal sensor readings |
| `FaceData` | `face_data_pb2` | Face detection bounding boxes |
| `MirrorSystem` | `mirror_system_pb2` | Mirror actuator state (C-array) |

---

## Primitive types

| Message | File | Fields |
|---|---|---|
| `Point2I` | `point2i_pb2` | `x: int32`, `y: int32` |
| `Point2F` | `point2f_pb2` | `x: float`, `y: float` |
| `Point3F` | `point3f_pb2` | `x: float`, `y: float`, `z: float` |
| `Range2F` | `range2f_pb2` | `min: float`, `max: float` |
| `RectangleI` | `rectanglei_pb2` | `x, y, width, height: int32` |
| `Matrix3x3F` | `matrix3x3f_pb2` | `x00`…`x22: float` (9 fields, row-major) |
| `Matrix4x4F` | `matrix4x4f_pb2` | `x00`…`x33: float` (16 fields, row-major) |
| `TimeStamp` | `time_stamp_pb2` | Unix timestamp |
| `CameraId` | `camera_id_pb2` | Enum 0–15 |
| `SensorType` | `sensor_type_pb2` | Sensor model enum |

---

## `sys.path` import resolution

The generated `_pb2.py` files use flat imports (`import camera_module_pb2`)
rather than package-relative imports. `shadow/proto/__init__.py` inserts the
`shadow/proto/` directory into `sys.path` at import time so these flat
imports resolve correctly.

Always import `shadow.proto` (or any `shadow.proto.*`) before any
`_pb2` cross-imports are triggered. In practice, importing `shadow._proto`
is sufficient, as it imports `shadow.proto` explicitly.
