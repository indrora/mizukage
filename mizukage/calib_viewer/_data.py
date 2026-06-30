"""Calibration data loading for the calib-view explorer."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class CalibData:
    """All calibration data for a lightcal directory, parsed at startup."""
    calib_dir: Path
    cameras: list[str]                        # ordered camera names present in calibration
    geometry: dict[str, list[dict[str, Any]]] # camera → list of geo dicts (one per block)
    color: dict[str, list[dict[str, Any]]]    # camera → list of color dicts per illuminant
    vignetting: dict[str, dict[str, Any]]     # camera → vignetting dict
    hot_pixels: dict[str, np.ndarray | None]  # camera → boolean bitmap or None
    hp_stats: dict[str, list[dict]]           # camera → list of measurement stats
    vst_entries: list[Any]                    # list[VSTEntry] from _calib
    black_level: float
    white_level: float
    device_model: str
    extrinsics: dict[str, dict]               # camera → {R, t, camera_loc, is_movable, mirror_info}
    calib_timestamp: str                       # "YYYY-MM-DD HH:MM:SS" from mc.time_stamp, or ""


_CAM_ORDER = [
    "A1", "A2", "A3", "A4", "A5",
    "B1", "B2", "B3", "B4", "B5",
    "C1", "C2", "C3", "C4", "C5", "C6",
]


def load_calib_data(calib_dir: Path) -> CalibData:
    """Parse calibration.lri and hotpixel.rec; return CalibData."""
    from mizukage._block import iter_blocks, BlockType
    import mizukage._proto as _proto
    from mizukage._calib import load_vst_model, load_hot_pixel_map
    from mizukage._types import CameraId
    from google.protobuf.json_format import MessageToDict
    from mizukage.proto import camera_id_pb2
    import struct, zlib

    cal_path = calib_dir / "calibration.lri"
    hp_path  = calib_dir / "hotpixel.rec"

    # ── Parse calibration.lri ─────────────────────────────────────────────────
    geometry: dict[str, list[dict]] = {}
    color: dict[str, list[dict]] = {}
    vignetting: dict[str, dict] = {}
    black_level = 42.0
    white_level = 1023.0
    device_model = ""
    calib_timestamp = ""

    if cal_path.exists():
        raw = cal_path.read_bytes()
        for block_start, hdr in iter_blocks(raw):
            if hdr.msg_type != BlockType.LIGHT_HEADER:
                continue
            proto_bytes = raw[
                block_start + hdr.msg_offset :
                block_start + hdr.msg_offset + hdr.msg_len
            ]
            if not proto_bytes:
                continue
            try:
                lh = _proto.parse_light_header(proto_bytes)
            except Exception:
                continue

            if not device_model and lh.HasField("device_model_name"):
                device_model = lh.device_model_name

            # Calibration timestamp (first mc that has one)
            if not calib_timestamp:
                for mc_ts in lh.module_calibration:
                    if mc_ts.HasField("time_stamp"):
                        ts = mc_ts.time_stamp
                        calib_timestamp = (
                            f"{ts.year}-{ts.month:02d}-{ts.day:02d}"
                            f" {ts.hour:02d}:{ts.minute:02d}:{ts.second:02d}"
                        )
                        break

            # Sensor characteristics
            for sd in lh.sensor_data:
                if sd.HasField("data"):
                    black_level = float(sd.data.black_level)
                    white_level = float(sd.data.white_level)
                    break

            for mc in lh.module_calibration:
                cam = camera_id_pb2.CameraID.Name(mc.camera_id)

                if mc.HasField("geometry"):
                    geo_d = MessageToDict(mc.geometry, preserving_proto_field_name=True)
                    geometry.setdefault(cam, []).append(geo_d)

                for cc in mc.color:
                    color.setdefault(cam, []).append(
                        MessageToDict(cc, preserving_proto_field_name=True)
                    )

                if mc.HasField("vignetting") and cam not in vignetting:
                    vignetting[cam] = MessageToDict(mc.vignetting, preserving_proto_field_name=True)

    # ── Extrinsics (rotation, translation, camera world positions) ───────────
    # Access proto directly — MessageToDict loses sub-message structure for
    # nested oneof fields like canonical / moveable_mirror.
    extrinsics: dict[str, dict] = {}

    if cal_path.exists():
        raw = cal_path.read_bytes()
        for block_start, hdr in iter_blocks(raw):
            if hdr.msg_type != BlockType.LIGHT_HEADER:
                continue
            proto_bytes = raw[
                block_start + hdr.msg_offset :
                block_start + hdr.msg_offset + hdr.msg_len
            ]
            if not proto_bytes:
                continue
            try:
                lh = _proto.parse_light_header(proto_bytes)
            except Exception:
                continue

            for mc in lh.module_calibration:
                cam = camera_id_pb2.CameraID.Name(mc.camera_id)
                # Only process each camera once, and only if it has geometry
                if cam in extrinsics or not mc.HasField("geometry"):
                    continue
                g = mc.geometry
                mirror_type = int(g.mirror_type)  # 0=NONE, 1=GLUED, 2=MOVABLE
                mirror_names = {0: "Fixed", 1: "Glued", 2: "Movable"}

                for b in g.per_focus_calibration:
                    if not b.HasField("extrinsics"):
                        continue
                    e = b.extrinsics
                    entry: dict = {
                        "mirror_type": mirror_names.get(mirror_type, str(mirror_type)),
                        "is_movable": mirror_type == 2,
                    }

                    if e.HasField("canonical"):
                        r = e.canonical.rotation
                        t = e.canonical.translation
                        R = [
                            [r.x00, r.x01, r.x02],
                            [r.x10, r.x11, r.x12],
                            [r.x20, r.x21, r.x22],
                        ]
                        tvec = [t.x, t.y, t.z]
                        # Camera world position = -R^T @ t
                        loc = [
                            -sum(R[j][i] * tvec[j] for j in range(3))
                            for i in range(3)
                        ]
                        entry.update({
                            "R": R,
                            "t": tvec,
                            "camera_loc": loc,
                            "mirror_info": None,
                        })

                    elif e.HasField("moveable_mirror"):
                        ms  = e.moveable_mirror.mirror_system
                        mam = e.moveable_mirror.mirror_actuator_mapping
                        loc = [
                            ms.real_camera_location.x,
                            ms.real_camera_location.y,
                            ms.real_camera_location.z,
                        ]
                        entry.update({
                            "R": None,
                            "t": None,
                            "camera_loc": loc,
                            "mirror_info": {
                                "rotation_axis": [
                                    ms.rotation_axis.x,
                                    ms.rotation_axis.y,
                                    ms.rotation_axis.z,
                                ],
                                "actuator_length_offset": mam.actuator_length_offset,
                                "actuator_length_scale":  mam.actuator_length_scale,
                                "mirror_angle_offset":    mam.mirror_angle_offset,
                                "mirror_angle_scale":     mam.mirror_angle_scale,
                            },
                        })

                    extrinsics[cam] = entry
                    break  # only one extrinsics bundle per camera

    # ── VST noise model ───────────────────────────────────────────────────────
    vst_entries = load_vst_model(calib_dir)

    # ── Hot pixels ────────────────────────────────────────────────────────────
    # Also parse measurement metadata from hotpixel.rec separately
    hp_bitmaps: dict[str, np.ndarray | None] = {}
    hp_stats: dict[str, list[dict]] = {}

    all_cams = sorted(
        set(geometry) | set(color) | set(vignetting),
        key=lambda c: _CAM_ORDER.index(c) if c in _CAM_ORDER else 99,
    )

    if hp_path.exists():
        # Reparse hotpixel.rec for stats (exposure, temp, gain per measurement)
        raw_hp = hp_path.read_bytes()
        from mizukage._block import iter_blocks, BlockType
        for block_start, hdr in iter_blocks(raw_hp):
            if hdr.msg_type != BlockType.LIGHT_HEADER:
                continue
            proto_bytes = raw_hp[
                block_start + hdr.msg_offset :
                block_start + hdr.msg_offset + hdr.msg_len
            ]
            if not proto_bytes:
                continue
            try:
                lh = _proto.parse_light_header(proto_bytes)
            except Exception:
                continue
            for mc in lh.module_calibration:
                if not mc.HasField("hot_pixel_map"):
                    continue
                from mizukage.proto import camera_id_pb2
                cam = camera_id_pb2.CameraID.Name(mc.camera_id)
                measurements = []
                for meas in mc.hot_pixel_map.data:
                    abs_offset = block_start + int(meas.data_offset)
                    blob = raw_hp[abs_offset : abs_offset + meas.data_size]
                    try:
                        _, _, _, width, height = struct.unpack_from("<IIIII", blob, 0)
                        bitmap_bytes = zlib.decompress(blob[20:])
                        n_hot = sum(bitmap_bytes)
                        total = len(bitmap_bytes)
                        measurements.append({
                            "width": width,
                            "height": height,
                            "hot_pixel_count": n_hot,
                            "hot_pixel_fraction": n_hot / total if total else 0.0,
                            "sensor_gain": float(meas.sensor_gain),
                            "sensor_temperature_c": float(meas.sensor_temparature),
                            "sensor_exposure_us": int(meas.data_exposure),
                        })
                    except Exception:
                        pass
                if measurements:
                    hp_stats[cam] = measurements
                    if cam not in all_cams:
                        all_cams.append(cam)
                        all_cams.sort(key=lambda c: _CAM_ORDER.index(c) if c in _CAM_ORDER else 99)

        # Load bitmaps (OR of all measurements) per camera
        for cam in all_cams:
            try:
                cam_id_int = _CAM_ORDER.index(cam)
                from mizukage._types import CameraId as CID
                cam_enum = CID(cam_id_int)
                hp_bitmaps[cam] = load_hot_pixel_map(calib_dir, cam_enum)
            except Exception:
                hp_bitmaps[cam] = None
    else:
        for cam in all_cams:
            hp_bitmaps[cam] = None

    return CalibData(
        calib_dir=calib_dir,
        cameras=all_cams,
        geometry=geometry,
        color=color,
        vignetting=vignetting,
        hot_pixels=hp_bitmaps,
        hp_stats=hp_stats,
        vst_entries=vst_entries,
        black_level=black_level,
        white_level=white_level,
        device_model=device_model,
        extrinsics=extrinsics,
        calib_timestamp=calib_timestamp,
    )


def mat3_values(mat_dict: dict, order: str = "row") -> list[float]:
    """Flatten a Matrix3x3F protobuf-dict to a 9-element row-major list."""
    return [mat_dict.get(f"x{r}{c}", 0.0) for r in range(3) for c in range(3)]
