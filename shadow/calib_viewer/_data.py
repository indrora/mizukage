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


_CAM_ORDER = [
    "A1", "A2", "A3", "A4", "A5",
    "B1", "B2", "B3", "B4", "B5",
    "C1", "C2", "C3", "C4", "C5", "C6",
]


def load_calib_data(calib_dir: Path) -> CalibData:
    """Parse calibration.lri and hotpixel.rec; return CalibData."""
    from shadow._block import iter_blocks, BlockType
    import shadow._proto as _proto
    from shadow._calib import load_vst_model, load_hot_pixel_map
    from shadow._types import CameraId
    from google.protobuf.json_format import MessageToDict
    from shadow.proto import camera_id_pb2
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
        from shadow._block import iter_blocks, BlockType
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
                from shadow.proto import camera_id_pb2
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
                from shadow._types import CameraId as CID
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
    )


def mat3_values(mat_dict: dict, order: str = "row") -> list[float]:
    """Flatten a Matrix3x3F protobuf-dict to a 9-element row-major list."""
    return [mat_dict.get(f"x{r}{c}", 0.0) for r in range(3) for c in range(3)]
