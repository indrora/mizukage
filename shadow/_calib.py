"""shadow._calib — shared calibration utilities for the image pipeline.

This module provides hot-pixel map loading and correction for use in the
export pipeline.  It is *not* a CLI module; for the calib inspection
command see shadow/cli/commands/calib.py.
"""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

from shadow._types import CameraId

# Name of the hot-pixel record file inside a lightcal directory.
_HOTPIXEL_REC = "hotpixel.rec"


def load_hot_pixel_map(calib_dir: Path, camera_id: CameraId) -> np.ndarray | None:
    """Load the hot-pixel bitmap for one camera from a lightcal directory.

    Returns a boolean uint8 array of shape (height, width) where 1=hot, or None
    if hotpixel.rec is absent or has no entry for this camera.

    When multiple measurements are present for the same camera (e.g. at
    different gain/exposure combinations), the bitmaps are OR-ed together so
    that a pixel hot in any calibration run is treated as hot.
    """
    hp_path = calib_dir / _HOTPIXEL_REC
    if not hp_path.exists():
        return None

    from shadow._block import iter_blocks, BlockType
    import shadow._proto as _proto

    data = hp_path.read_bytes()
    combined: np.ndarray | None = None

    for block_start, hdr in iter_blocks(data):
        if hdr.msg_type != BlockType.LIGHT_HEADER:
            continue
        proto_bytes = data[
            block_start + hdr.msg_offset:
            block_start + hdr.msg_offset + hdr.msg_len
        ]
        if not proto_bytes:
            continue
        try:
            msg = _proto.parse_light_header(proto_bytes)
        except Exception:
            continue

        for mc in msg.module_calibration:
            if not mc.HasField("hot_pixel_map"):
                continue
            # mc.camera_id is an integer matching the CameraId IntEnum values.
            if int(mc.camera_id) != int(camera_id):
                continue

            hpm = mc.hot_pixel_map
            for meas in hpm.data:
                abs_offset = block_start + int(meas.data_offset)
                raw = data[abs_offset: abs_offset + meas.data_size]
                try:
                    # 20-byte header: timestamp, padding, comp_size, width, height.
                    _, _, _comp_size, width, height = struct.unpack_from(
                        "<IIIII", raw, 0
                    )
                    bitmap_bytes = zlib.decompress(raw[20:])
                    bitmap = np.frombuffer(bitmap_bytes, dtype=np.uint8).reshape(
                        height, width
                    )
                    # Normalise: any non-zero byte counts as a hot pixel.
                    bitmap = (bitmap != 0).astype(np.uint8)
                    if combined is None:
                        combined = bitmap
                    else:
                        # OR bitmaps: pixel is hot if it fires in any run.
                        np.bitwise_or(combined, bitmap, out=combined)
                except Exception:
                    continue

    return combined


def apply_hot_pixel_correction(
    bayer: np.ndarray,   # uint16 (H, W), raw Bayer data
    bitmap: np.ndarray,  # uint8 (H, W), 1 = hot pixel
    r_row: int,          # 0 or 1 — row offset of red pixel in 2×2 Bayer tile
    r_col: int,          # 0 or 1 — col offset of red pixel in 2×2 Bayer tile
) -> np.ndarray:
    """Replace hot pixels with bilinear mean of same-colour-channel Bayer neighbours.

    For a hot pixel at (row, col):
    - Its colour channel is determined by (row % 2, col % 2) relative to the
      Bayer tile (r_row, r_col) — the arguments are accepted for forward
      compatibility but the ±2 neighbour offsets apply equally to every channel.
    - Same-channel axis-aligned neighbours are at offsets (±2, 0) and (0, ±2).
      Diagonal ±2 offsets are *not* used.
    - Only neighbours within the image boundary contribute to the average.
    - Results are rounded and clamped to [0, 65535].

    Returns a new uint16 array (copy, not in-place).
    """
    out = bayer.copy()
    H, W = bayer.shape

    hot_rows, hot_cols = np.nonzero(bitmap)
    if hot_rows.size == 0:
        return out

    # Four axis-aligned same-channel neighbour offsets.
    offsets = ((-2, 0), (2, 0), (0, -2), (0, 2))

    # Accumulate neighbour values in float64 to avoid uint16 overflow.
    sum_vals = np.zeros(hot_rows.size, dtype=np.float64)
    count    = np.zeros(hot_rows.size, dtype=np.int32)

    for dr, dc in offsets:
        nr = hot_rows + dr
        nc = hot_cols + dc
        # Include only indices that fall within the image.
        valid = (nr >= 0) & (nr < H) & (nc >= 0) & (nc < W)
        if not np.any(valid):
            continue
        sum_vals[valid] += bayer[nr[valid], nc[valid]].astype(np.float64)
        count[valid] += 1

    # For pixels with at least one valid neighbour, replace with rounded mean;
    # leave the original value if somehow no neighbour is available (very small
    # image edge case — should not occur on real L16 sensors).
    has_nb = count > 0
    safe_count = np.where(has_nb, count, 1).astype(np.float64)
    averaged   = np.round(sum_vals / safe_count).astype(np.int64)
    corrected  = np.where(
        has_nb,
        np.clip(averaged, 0, 65535),
        bayer[hot_rows, hot_cols].astype(np.int64),
    ).astype(np.uint16)

    out[hot_rows, hot_cols] = corrected
    return out
