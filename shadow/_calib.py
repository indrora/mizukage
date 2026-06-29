"""shadow._calib — shared calibration utilities for the image pipeline.

Provides:
  * load_hot_pixel_map / apply_hot_pixel_correction — hotpixel.rec support
  * VSTEntry / load_vst_model / compute_scalar_sigma — VST noise model from calibration.lri

The VST (Variance-Stabilising Transform) noise model records, for each gain
setting, a per-channel linear model:

    variance(signal) = a * signal + b

This module produces a single scalar sigma representative of the whole image
(green channel at mid-signal), suitable for BM3D, bilateral, and DRUNet.

This module is *not* a CLI module; for the calib inspection command see
shadow/cli/commands/calib.py.
"""
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from shadow._types import CameraId

# Name of the hot-pixel record file inside a lightcal directory.
_HOTPIXEL_REC = "hotpixel.rec"


# ── Hot-pixel correction ───────────────────────────────────────────────────────

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
            block_start + hdr.msg_offset :
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
                raw = data[abs_offset : abs_offset + meas.data_size]
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
    # leave the original value if somehow no neighbour is available.
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


# ── VST noise model ────────────────────────────────────────────────────────────

@dataclass
class VSTEntry:
    """One row of the factory VST noise model table.

    ``gain_x100`` matches the proto ``gain`` field: values 100–775, step 25.
    Per-channel linear noise-variance model: variance = a * signal + b.
    """
    gain_x100: int        # gain * 100 (100..775, step 25)
    # Per-channel linear model coefficients
    r_a: float;  r_b: float
    g_a: float;  g_b: float
    b_a: float;  b_b: float


def load_vst_model(calib_dir: Path) -> list[VSTEntry]:
    """Parse the VST noise model from calibration.lri in a lightcal directory.

    Reads ``<calib_dir>/calibration.lri`` as an LELR block stream, finds the
    first ``SensorCharacterization`` proto with a non-empty ``vst_model``
    repeated field, and returns the entries sorted by ``gain_x100``.

    Returns an empty list when the file is absent, unreadable, or contains no
    VST entries — callers must treat an empty return as "model unavailable" and
    fall back to a default sigma.
    """
    calib_path = calib_dir / "calibration.lri"
    if not calib_path.exists():
        return []

    from shadow._block import iter_blocks, BlockType
    import shadow._proto as _proto

    try:
        data = calib_path.read_bytes()
    except OSError:
        return []

    for block_start, hdr in iter_blocks(data):
        if hdr.msg_type != BlockType.LIGHT_HEADER:
            continue
        proto_bytes = data[
            block_start + hdr.msg_offset :
            block_start + hdr.msg_offset + hdr.msg_len
        ]
        if not proto_bytes:
            continue
        try:
            lh = _proto.parse_light_header(proto_bytes)
        except Exception:
            continue

        for sd in lh.sensor_data:
            # SensorData wraps the actual SensorCharacterization in a 'data' sub-field.
            if not sd.HasField("data"):
                continue
            raw_entries = list(sd.data.vst_model)
            if not raw_entries:
                continue
            entries: list[VSTEntry] = []
            for vst in raw_entries:
                entries.append(VSTEntry(
                    gain_x100=int(vst.gain),
                    r_a=float(vst.red.a),   r_b=float(vst.red.b),
                    g_a=float(vst.green.a), g_b=float(vst.green.b),
                    b_a=float(vst.blue.a),  b_b=float(vst.blue.b),
                ))
            # Sort ascending so nearest-entry lookup always works correctly.
            return sorted(entries, key=lambda e: e.gain_x100)

    return []


def compute_scalar_sigma(
    vst_model: list[VSTEntry],
    analog_gain: float,
    white_level: int = 981,  # typical L16: 1023 raw max − 42 black level
) -> float:
    """Derive a representative scalar sigma from the factory VST noise model.

    Looks up the VST entry nearest to ``analog_gain * 100`` (snapping to the
    25-unit grid), then estimates sigma at mid-signal using the green channel
    (highest photon count and therefore most representative for luma-weighted
    denoisers), normalised to the [0, 1] floating-point range used by BM3D /
    bilateral / DRUNet:

        sigma = sqrt(g_a * (white_level / 2) + g_b) / white_level

    The result is clamped to [0.01, 0.30] so an unusually large or negative
    variance estimate can never produce a harmful sigma value.

    Returns 0.05 (a safe conservative default) when ``vst_model`` is empty.
    """
    if not vst_model:
        return 0.05

    # Snap the capture gain to the nearest 25-unit grid point, then clamp to
    # the model's actual range so we never extrapolate beyond the table.
    target = round(analog_gain * 100 / 25) * 25
    target = max(vst_model[0].gain_x100, min(vst_model[-1].gain_x100, target))
    entry = min(vst_model, key=lambda e: abs(e.gain_x100 - target))

    mid_signal = white_level / 2.0
    variance = entry.g_a * mid_signal + entry.g_b
    # Guard against numerically negative variance from near-zero b coefficients.
    if variance <= 0.0:
        return 0.05
    sigma_raw = variance ** 0.5 / white_level  # normalise to [0, 1]
    return max(0.01, min(0.30, sigma_raw))
