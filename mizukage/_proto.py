"""Proto-to-types bridge.

This is the only module in mizukage that imports from mizukage.proto.*.
It converts raw protobuf objects into mizukage._types instances.
"""
from __future__ import annotations

# Importing mizukage.proto triggers the sys.path patch in mizukage/proto/__init__.py
# so that all inter-_pb2 flat imports resolve correctly.
import mizukage.proto  # noqa: F401 — side-effect import for sys.path patch

from mizukage.proto import (
    lightheader_pb2,
    view_preferences_pb2,
    gps_data_pb2,
)
from mizukage._types import (
    AwbGains,
    AwbMode,
    BayerPattern,
    CameraId,
    CaptureMetadata,
    ColorProfile,
    GpsData,
    HdrMode,
    Illuminant,
    ModuleCapture,
    Orientation,
    SceneMode,
    SensorModel,
)


# ── Parsers ────────────────────────────────────────────────────────────────────

def parse_light_header(data: bytes) -> lightheader_pb2.LightHeader:
    msg = lightheader_pb2.LightHeader()
    msg.ParseFromString(data)
    return msg


def parse_view_preferences(data: bytes) -> view_preferences_pb2.ViewPreferences:
    msg = view_preferences_pb2.ViewPreferences()
    msg.ParseFromString(data)
    return msg


def parse_gps_data_proto(data: bytes) -> gps_data_pb2.GPSData:
    msg = gps_data_pb2.GPSData()
    msg.ParseFromString(data)
    return msg


# ── Converters ─────────────────────────────────────────────────────────────────

def sensor_id_map_from_hw_info(hw_info) -> dict[CameraId, SensorModel]:
    """Build CameraId→SensorModel from an HwInfo proto message (may be None)."""
    result: dict[CameraId, SensorModel] = {}
    if hw_info is None:
        return result
    for cam_hw in hw_info.camera:
        try:
            cam_id = CameraId(cam_hw.id)
            sensor = SensorModel(cam_hw.sensor)
            result[cam_id] = sensor
        except (ValueError, TypeError):
            pass
    return result


def color_profiles_from_calibrations(cals) -> dict[CameraId, list[ColorProfile]]:
    """Extract ColorProfile list per CameraId from repeated FactoryModuleCalibration."""
    result: dict[CameraId, list[ColorProfile]] = {}
    for cal in cals:
        try:
            cam_id = CameraId(cal.camera_id)
        except (ValueError, TypeError):
            continue
        profiles: list[ColorProfile] = []
        for color in cal.color:
            try:
                illuminant = Illuminant(color.type)
            except ValueError:
                illuminant = Illuminant.UNKNOWN
            fw = color.forward_matrix
            cm = color.color_matrix
            profiles.append(ColorProfile(
                camera_id=cam_id,
                illuminant=illuminant,
                forward_matrix=(
                    fw.x00, fw.x01, fw.x02,
                    fw.x10, fw.x11, fw.x12,
                    fw.x20, fw.x21, fw.x22,
                ),
                color_matrix=(
                    cm.x00, cm.x01, cm.x02,
                    cm.x10, cm.x11, cm.x12,
                    cm.x20, cm.x21, cm.x22,
                ),
                rg_ratio=color.rg_ratio,
                bg_ratio=color.bg_ratio,
            ))
        if profiles:
            result.setdefault(cam_id, [])
            result[cam_id].extend(profiles)
    return result


def awb_gains_from_vp(vp) -> AwbGains | None:
    """Extract AwbGains from a ViewPreferences proto (or None if absent)."""
    if not vp.HasField("awb_gains"):
        return None
    g = vp.awb_gains
    return AwbGains(r=g.r, gr=g.g_r, gb=g.g_b, b=g.b)


def gps_from_proto(gps_proto) -> GpsData | None:
    """Convert a GPSData proto into a GpsData dataclass (or None if empty)."""
    if gps_proto is None:
        return None
    lat = gps_proto.latitude if gps_proto.HasField("latitude") else None
    lon = gps_proto.longitude if gps_proto.HasField("longitude") else None
    if lat is None or lon is None:
        return None
    alt = gps_proto.altitude.value if gps_proto.HasField("altitude") else None
    hdg = gps_proto.heading.value if gps_proto.HasField("heading") else None
    spd = gps_proto.speed if gps_proto.HasField("speed") else None
    return GpsData(latitude=lat, longitude=lon, altitude_m=alt, heading=hdg, speed=spd)


def bayer_pos_from_sbro(sbro) -> tuple[int, int] | None:
    """Convert sensor_bayer_red_override Point2I to (r_row, r_col).

    Returns None for mono sensors (sbro absent or sentinel -1 values).
    """
    if sbro is None:
        return None
    bx = sbro.x
    by = sbro.y
    if bx < 0 or by < 0:
        return None
    return (by % 2, bx % 2)


def black_level_from_sensor_data(sensor_data_list, default: float = 64.0) -> float:
    """Extract black_level from the first SensorData entry, or return default."""
    for sd in sensor_data_list:
        if sd.HasField("data"):
            return float(sd.data.black_level)
    return default


def module_capture_from_proto(mod) -> ModuleCapture:
    """Convert a CameraModule proto to a ModuleCapture dataclass."""
    try:
        cam_id = CameraId(mod.id)
    except ValueError:
        cam_id = CameraId.A1  # fallback; shouldn't happen

    mirror_pos: int | None = None
    if mod.HasField("mirror_position"):
        mirror_pos = mod.mirror_position

    digital_gain: float | None = None
    if mod.HasField("sensor_digital_gain"):
        digital_gain = float(mod.sensor_digital_gain)

    return ModuleCapture(
        camera_id=cam_id,
        enabled=mod.is_enabled,
        lens_position=mod.lens_position,
        mirror_position=mirror_pos,
        analog_gain=float(mod.sensor_analog_gain),
        exposure_ns=int(mod.sensor_exposure),
        digital_gain=digital_gain,
        flip_h=mod.sensor_is_horizontal_flip,
        flip_v=mod.sensor_is_vertical_flip,
    )


def update_metadata_from_light_header(meta: CaptureMetadata, lh) -> None:
    """Merge fields from a LightHeader proto into a CaptureMetadata (in-place)."""
    if meta.image_id == (0, 0):
        low = lh.image_unique_id_low if lh.HasField("image_unique_id_low") else 0
        high = lh.image_unique_id_high if lh.HasField("image_unique_id_high") else 0
        meta.image_id = (int(low), int(high))

    if meta.focal_length_mm is None and lh.HasField("image_focal_length"):
        meta.focal_length_mm = int(lh.image_focal_length)

    if meta.reference_camera is None and lh.HasField("image_reference_camera"):
        try:
            meta.reference_camera = CameraId(lh.image_reference_camera)
        except ValueError:
            pass

    if not meta.device_model and lh.HasField("device_model_name"):
        meta.device_model = lh.device_model_name

    if not meta.firmware_version and lh.HasField("device_fw_version"):
        meta.firmware_version = lh.device_fw_version

    # ViewPreferences embedded in LightHeader (field 19)
    if lh.HasField("view_preferences"):
        update_metadata_from_view_prefs(meta, lh.view_preferences)

    # Per-module capture settings
    for mod in lh.modules:
        try:
            cam_id = CameraId(mod.id)
        except ValueError:
            continue
        if cam_id not in meta.modules and mod.HasField("sensor_data_surface"):
            meta.modules[cam_id] = module_capture_from_proto(mod)

    # GPS (field 25 in LightHeader)
    if meta.gps is None and lh.HasField("gps_data"):
        meta.gps = gps_from_proto(lh.gps_data)

    # IMU-derived orientation (field 23); ViewPreferences takes priority if set
    if meta.orientation is None:
        meta.orientation = imu_orientation_from_light_header(lh)


def imu_orientation_from_light_header(lh) -> Orientation | None:
    """Infer device-hold orientation from IMU accelerometer samples.

    The L16 accelerometer convention (empirically derived):
        +Y dominant → landscape, natural (Y axis points down in landscape hold)
        −Y dominant → landscape, upside-down
        −X dominant → portrait, top-left  (device rotated CCW from landscape)
        +X dominant → portrait, top-right (device rotated CW  from landscape)
        Z  dominant → camera lying flat; orientation undefined → None

    Returns the display correction to apply *after* the 180° sensor-mount fix.
    """
    imu_list = list(lh.imu_data)
    if not imu_list:
        return None

    ax_sum = ay_sum = az_sum = 0.0
    count = 0
    for imu in imu_list:
        for sample in imu.accelerometer:
            ax_sum += sample.data.x
            ay_sum += sample.data.y
            az_sum += sample.data.z
            count += 1
    if count == 0:
        return None

    ax = ax_sum / count
    ay = ay_sum / count
    az = az_sum / count

    abs_ax, abs_ay, abs_az = abs(ax), abs(ay), abs(az)
    if abs_az >= abs_ax and abs_az >= abs_ay:
        return None  # flat on a surface
    if abs_ay >= abs_ax:
        return Orientation.NORMAL if ay > 0 else Orientation.ROT180
    return Orientation.ROT90_CCW if ax < 0 else Orientation.ROT90_CW


def update_metadata_from_view_prefs(meta: CaptureMetadata, vp) -> None:
    """Merge ViewPreferences proto fields into CaptureMetadata (in-place)."""
    if meta.awb_gains is None:
        meta.awb_gains = awb_gains_from_vp(vp)

    if meta.awb_mode is None and vp.HasField("awb_mode"):
        try:
            meta.awb_mode = AwbMode(vp.awb_mode)
        except ValueError:
            pass

    if meta.hdr_mode is None and vp.HasField("hdr_mode"):
        try:
            meta.hdr_mode = HdrMode(vp.hdr_mode)
        except ValueError:
            pass

    if meta.scene_mode is None and vp.HasField("scene_mode"):
        try:
            meta.scene_mode = SceneMode(vp.scene_mode)
        except ValueError:
            pass

    if meta.on_tripod is None and vp.HasField("is_on_tripod"):
        meta.on_tripod = bool(vp.is_on_tripod)

    if meta.orientation is None and vp.HasField("orientation"):
        try:
            meta.orientation = Orientation(vp.orientation)
        except ValueError:
            pass


def update_metadata_from_gps_block(meta: CaptureMetadata, gps_proto) -> None:
    """Merge a top-level GPSData block proto into CaptureMetadata (in-place)."""
    if meta.gps is None:
        meta.gps = gps_from_proto(gps_proto)
