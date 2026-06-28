"""Pure data types for the shadow library.

No I/O, no numpy. All other modules import from here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class CameraId(IntEnum):
    A1 = 0
    A2 = 1
    A3 = 2
    A4 = 3
    A5 = 4
    B1 = 5
    B2 = 6
    B3 = 7
    B4 = 8
    B5 = 9
    C1 = 10
    C2 = 11
    C3 = 12
    C4 = 13
    C5 = 14
    C6 = 15

    @classmethod
    def from_name(cls, name: str) -> "CameraId":
        """Parse 'A1', 'b3', 'C6' etc. into a CameraId."""
        return cls[name.upper()]

    @property
    def array(self) -> str:
        """'A', 'B', or 'C' — which focal-length array this camera belongs to."""
        return self.name[0]

    @property
    def index(self) -> int:
        """1-based index within the array (1–5 for A/B, 1–6 for C)."""
        return int(self.name[1:])


class SensorModel(IntEnum):
    UNKNOWN = 0
    AR835 = 1
    AR1335 = 2
    AR1335_MONO = 3
    IMX386 = 4
    IMX386_MONO = 5

    @property
    def is_mono(self) -> bool:
        return self in (SensorModel.AR1335_MONO, SensorModel.IMX386_MONO)


class RawFormat(IntEnum):
    BAYER_JPEG = 0
    PACKED_10BPP = 7
    PACKED_12BPP = 8
    PACKED_14BPP = 9


class BayerPattern(IntEnum):
    """CFA position of the R pixel within the 2×2 Bayer tile.

    Encoded as (r_col | (r_row << 1)):
        RGGB = R at (row=0, col=0)
        GRBG = R at (row=0, col=1)
        GBRG = R at (row=1, col=0)
        BGGR = R at (row=1, col=1)   ← most common on L16 AR1335
    """
    RGGB = 0
    GRBG = 1
    GBRG = 2
    BGGR = 3

    @property
    def r_row(self) -> int:
        return self.value >> 1

    @property
    def r_col(self) -> int:
        return self.value & 1


class Illuminant(IntEnum):
    A = 0
    D50 = 1
    D65 = 2
    D75 = 3
    F2 = 4
    F7 = 5
    F11 = 6
    TL84 = 7
    UNKNOWN = 99


class HdrMode(IntEnum):
    NONE = 0
    DEFAULT = 1
    NATURAL = 2
    SURREAL = 3


class SceneMode(IntEnum):
    PORTRAIT = 0
    LANDSCAPE = 1
    SPORT = 2
    MACRO = 3
    NIGHT = 4
    NONE = 5


class AwbMode(IntEnum):
    AUTO = 0
    DAYLIGHT = 1
    SHADE = 2
    CLOUDY = 3
    TUNGSTEN = 4
    FLUORESCENT = 5
    FLASH = 6
    CUSTOM = 7
    KELVIN = 8


class Orientation(IntEnum):
    NORMAL = 0
    ROT90_CW = 1
    ROT90_CCW = 2
    ROT90_CW_VFLIP = 3
    ROT90_CCW_VFLIP = 4
    VFLIP = 5
    HFLIP = 6
    ROT180 = 7


@dataclass(slots=True)
class AwbGains:
    """White-balance channel gains from the camera's AWB algorithm."""
    r: float
    gr: float
    gb: float
    b: float


@dataclass(slots=True)
class ColorProfile:
    """Per-camera, per-illuminant color calibration from factory data."""
    camera_id: CameraId
    illuminant: Illuminant
    forward_matrix: tuple[float, ...]  # 9 floats, row-major (RGB→XYZ)
    color_matrix: tuple[float, ...]    # 9 floats, row-major
    rg_ratio: float
    bg_ratio: float


@dataclass(slots=True)
class ModuleCapture:
    """Per-camera capture settings recorded at the moment of exposure."""
    camera_id: CameraId
    enabled: bool
    lens_position: int
    mirror_position: int | None
    analog_gain: float
    exposure_ns: int
    digital_gain: float | None
    flip_h: bool
    flip_v: bool


@dataclass(slots=True)
class GpsData:
    """GPS position recorded at capture time."""
    latitude: float
    longitude: float
    altitude_m: float | None
    heading: float | None
    speed: float | None


@dataclass
class CaptureMetadata:
    """Top-level metadata for an LRI capture."""
    image_id: tuple[int, int]           # (unique_id_low, unique_id_high)
    focal_length_mm: int | None
    reference_camera: CameraId | None
    device_model: str
    firmware_version: str
    awb_gains: AwbGains | None
    awb_mode: AwbMode | None
    hdr_mode: HdrMode | None
    scene_mode: SceneMode | None
    on_tripod: bool | None
    gps: GpsData | None
    modules: dict[CameraId, ModuleCapture] = field(default_factory=dict)
