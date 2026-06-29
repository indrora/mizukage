"""LriFile — main entry point for reading Light L16 LRI files."""
from __future__ import annotations

from pathlib import Path

from shadow._block import BlockType, iter_blocks
from shadow._image import RawImage
from shadow._types import (
    CameraId,
    CaptureMetadata,
    ColorProfile,
    RawFormat,
    SensorModel,
)
import shadow._proto as _proto


def _make_empty_metadata() -> CaptureMetadata:
    return CaptureMetadata(
        image_id=(0, 0),
        focal_length_mm=None,
        reference_camera=None,
        device_model="",
        firmware_version="",
        awb_gains=None,
        awb_mode=None,
        hdr_mode=None,
        scene_mode=None,
        on_tripod=None,
        gps=None,
    )


class LriFile:
    """Reader for a Light L16 LRI (Light Raw Image) file.

    Typical usage::

        lri = LriFile.open("photo.lri")
        print(lri.metadata.focal_length_mm)
        ref = lri.reference_image
        ref.to_png("reference.png")
    """

    def __init__(
        self,
        images: list[RawImage],
        metadata: CaptureMetadata,
        color_profiles: dict[CameraId, list[ColorProfile]],
    ) -> None:
        self._images = images
        self._metadata = metadata
        self._color_profiles = color_profiles

    @classmethod
    def open(cls, path: str | Path) -> "LriFile":
        """Open an LRI file and parse all LELR blocks.

        The entire file is read into memory once; image data is accessed
        via views into that buffer without additional copies.
        """
        data = Path(path).read_bytes()
        return cls._parse(data)

    @classmethod
    def _parse(cls, data: bytes) -> "LriFile":
        metadata = _make_empty_metadata()

        # images_by_cam: deduplicated per camera
        # PACKED_*: first occurrence wins.
        # BAYER_JPEG: keep the one with the lowest abs data offset.
        images_by_cam: dict[CameraId, RawImage] = {}
        color_profiles_by_cam: dict[CameraId, list[ColorProfile]] = {}
        sensor_id_map: dict[CameraId, SensorModel] = {}
        black_level: float = 64.0  # updated when sensor_data is found

        for block_start, hdr in iter_blocks(data):
            proto_bytes = data[
                block_start + hdr.msg_offset : block_start + hdr.msg_offset + hdr.msg_len
            ]
            if not proto_bytes:
                continue

            try:
                match hdr.msg_type:
                    case BlockType.LIGHT_HEADER:
                        lh = _proto.parse_light_header(proto_bytes)

                        # Sensor model lookup (HwInfo, field 18)
                        if lh.HasField("hw_info"):
                            sensor_id_map.update(
                                _proto.sensor_id_map_from_hw_info(lh.hw_info)
                            )

                        # Per-sensor black level (sensor_data, field 16)
                        bl = _proto.black_level_from_sensor_data(lh.sensor_data)
                        if bl != 64.0:
                            black_level = bl

                        # Factory color calibration (module_calibration, field 13)
                        for cam_id, profiles in _proto.color_profiles_from_calibrations(
                            lh.module_calibration
                        ).items():
                            color_profiles_by_cam.setdefault(cam_id, [])
                            color_profiles_by_cam[cam_id].extend(profiles)

                        # Per-module image data (modules, field 12)
                        for mod in lh.modules:
                            _process_module(
                                mod, block_start, data, images_by_cam, black_level
                            )

                        # Capture-level metadata
                        _proto.update_metadata_from_light_header(metadata, lh)

                    case BlockType.VIEW_PREFERENCES:
                        vp = _proto.parse_view_preferences(proto_bytes)
                        _proto.update_metadata_from_view_prefs(metadata, vp)

                    case BlockType.GPS_DATA:
                        gps_proto = _proto.parse_gps_data_proto(proto_bytes)
                        _proto.update_metadata_from_gps_block(metadata, gps_proto)

            except Exception:
                # Skip corrupt or unrecognised blocks rather than crashing
                continue

        # Back-fill sensor models and color profiles now that all blocks are read
        images: list[RawImage] = sorted(
            images_by_cam.values(), key=lambda img: img.camera_id
        )
        for img in images:
            if img.sensor_model == SensorModel.UNKNOWN and img.camera_id in sensor_id_map:
                img.sensor_model = sensor_id_map[img.camera_id]
            if img.camera_id in color_profiles_by_cam:
                img.color_profiles = color_profiles_by_cam[img.camera_id]
            img.awb_gains = metadata.awb_gains  # same gains apply to all modules

        return cls(images, metadata, color_profiles_by_cam)

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def images(self) -> list[RawImage]:
        """All active camera module images, sorted by CameraId."""
        return list(self._images)

    @property
    def metadata(self) -> CaptureMetadata:
        """Capture-level metadata (focal length, AWB, GPS, etc.)."""
        return self._metadata

    @property
    def color_profiles(self) -> dict[CameraId, list[ColorProfile]]:
        """Factory color calibration profiles, keyed by camera."""
        return dict(self._color_profiles)

    @property
    def reference_image(self) -> RawImage | None:
        """The camera designated as the reference by the firmware."""
        ref = self._metadata.reference_camera
        if ref is None:
            return None
        return self.image_for_camera(ref)

    def image_for_camera(self, camera: CameraId | str | int) -> RawImage | None:
        """Return the RawImage for a specific camera, or None if not present."""
        if isinstance(camera, str):
            try:
                camera = CameraId.from_name(camera)
            except KeyError:
                return None
        elif isinstance(camera, int):
            camera = CameraId(camera)
        return next((img for img in self._images if img.camera_id == camera), None)

    def images_by_camera(self) -> dict[CameraId, RawImage]:
        """Return a dict mapping CameraId to RawImage."""
        return {img.camera_id: img for img in self._images}


def _process_module(
    mod,
    block_start: int,
    data: bytes,
    images_by_cam: dict[CameraId, RawImage],
    black_level: float,
) -> None:
    """Extract image data from a CameraModule proto and add to images_by_cam."""
    if not mod.HasField("sensor_data_surface"):
        return

    try:
        cam_id = CameraId(mod.id)
    except ValueError:
        return

    surf = mod.sensor_data_surface
    try:
        fmt = RawFormat(surf.format)
    except ValueError:
        return  # unsupported format

    abs_data_offset = block_start + int(surf.data_offset)
    width = surf.size.x
    height = surf.size.y
    stride = int(surf.row_stride)

    # Deduplication:
    # PACKED_*: first occurrence per camera wins.
    # BAYER_JPEG: keep lowest abs_data_offset (= primary/shortest exposure).
    if cam_id in images_by_cam:
        existing = images_by_cam[cam_id]
        if fmt != RawFormat.BAYER_JPEG:
            return  # packed format already captured
        if abs_data_offset >= existing._data_offset:
            return  # keep the earlier (primary) JPEG

    # Parse sensor_bayer_red_override
    sbro = mod.sensor_bayer_red_override if mod.HasField("sensor_bayer_red_override") else None
    bayer_pos = _proto.bayer_pos_from_sbro(sbro)

    if bayer_pos is not None:
        r_row, r_col = bayer_pos
    else:
        # Default: BGGR (most common L16 AR1335 pattern)
        r_row, r_col = (1, 1)
        # Actually None for mono sensors
        if mod.HasField("sensor_bayer_red_override"):
            r_row = r_col = None  # type: ignore[assignment]

    images_by_cam[cam_id] = RawImage(
        camera_id=cam_id,
        sensor_model=SensorModel.UNKNOWN,  # back-filled after all blocks
        width=int(width),
        height=int(height),
        raw_format=fmt,
        bayer_r_row=r_row,
        bayer_r_col=r_col,
        analog_gain=float(mod.sensor_analog_gain),
        exposure_ns=int(mod.sensor_exposure),
        digital_gain=float(mod.sensor_digital_gain) if mod.HasField("sensor_digital_gain") else None,
        flip_h=bool(mod.sensor_is_horizontal_flip),
        flip_v=bool(mod.sensor_is_vertical_flip),
        _file_bytes=data,
        _data_offset=abs_data_offset,
        _row_stride=stride,
        _black_level=black_level,
    )
