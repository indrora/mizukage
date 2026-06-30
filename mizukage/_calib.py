"""shadow._calib — shared calibration utilities for the image pipeline.

Provides:
  * load_hot_pixel_map / apply_hot_pixel_correction — hotpixel.rec support
  * VSTEntry / load_vst_model / compute_scalar_sigma — VST noise model from calibration.lri
  * DistortionParams / load_distortion_params / undistort_image — radial lens undistortion

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

from mizukage._types import CameraId

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

    from mizukage._block import iter_blocks, BlockType
    import mizukage._proto as _proto

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

    from mizukage._block import iter_blocks, BlockType
    import mizukage._proto as _proto

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


# ── Vignetting correction ──────────────────────────────────────────────────────

def load_vignetting_grid(calib_dir: Path, camera_id: CameraId) -> np.ndarray | None:
    """Load the factory vignetting correction grid for one camera.

    Returns float32 (13, 17) array where each value is a multiplicative
    correction factor (1.0 = no correction, ~3.75 at corners).
    Returns None if calibration.lri is absent or has no entry for this camera.

    For cameras with multiple hall-code entries (movable-mirror C-array),
    returns the first entry (hall_code closest to 0 / rest position).
    """
    calib_path = calib_dir / "calibration.lri"
    if not calib_path.exists():
        return None

    from mizukage._block import iter_blocks, BlockType
    import mizukage._proto as _proto

    try:
        data = calib_path.read_bytes()
    except OSError:
        return None

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

        for mc in lh.module_calibration:
            # mc.camera_id is an integer matching the CameraId IntEnum values.
            if int(mc.camera_id) != int(camera_id):
                continue
            if not mc.HasField("vignetting"):
                continue

            # mc.vignetting is a VignettingCharacterization;
            # mc.vignetting.vignetting is a repeated MirrorVignettingModel.
            mirror_vigs = list(mc.vignetting.vignetting)
            if not mirror_vigs:
                continue

            # Use the first entry (hall_code=0 / rest position for C-array cameras;
            # fixed cameras have exactly one entry).
            mv = mirror_vigs[0]
            vm = mv.vignetting  # VignettingModel
            w = int(vm.width)
            h = int(vm.height)
            raw_data = list(vm.data)

            if w <= 0 or h <= 0 or len(raw_data) != w * h:
                continue

            grid = np.array(raw_data, dtype=np.float32).reshape(h, w)
            return grid

    return None


def apply_vignetting_correction(bayer_f32: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Multiply a float32 Bayer array by a bilinearly-interpolated vignetting grid.

    bayer_f32: float32 (H, W) — Bayer plane, already converted to float
    grid:      float32 (grid_h, grid_w) — correction factors from load_vignetting_grid

    The grid covers the full sensor; it is stretched to match (H, W) using
    bilinear interpolation via scipy.ndimage.zoom when available, or a
    pure-numpy 2-axis linear interpolation otherwise.

    Returns a new float32 array of the same shape.
    """
    H, W = bayer_f32.shape
    grid_h, grid_w = grid.shape

    # Fast path: no rescaling needed (grid already matches sensor size).
    if grid_h == H and grid_w == W:
        return bayer_f32 * grid

    # Preferred path: scipy zoom (bilinear, order=1).
    try:
        from scipy.ndimage import zoom as _zoom
        scale_h = H / grid_h
        scale_w = W / grid_w
        correction = _zoom(grid, (scale_h, scale_w), order=1)
    except ImportError:
        # Fallback: pure-numpy bilinear stretch via np.interp applied axis-by-axis.
        # 1. Stretch rows: interpolate each column along the H axis.
        row_coords = np.linspace(0.0, grid_h - 1, H)
        col_coords = np.linspace(0.0, grid_w - 1, W)

        # Interpolate along axis 0 (rows) for each column.
        grid_row_idx = np.arange(grid_h, dtype=np.float32)
        stretched_rows = np.stack(
            [np.interp(row_coords, grid_row_idx, grid[:, c]) for c in range(grid_w)],
            axis=1,
        )  # shape: (H, grid_w)

        # Interpolate along axis 1 (cols) for each row of the row-stretched result.
        grid_col_idx = np.arange(grid_w, dtype=np.float32)
        correction = np.stack(
            [np.interp(col_coords, grid_col_idx, stretched_rows[r, :]) for r in range(H)],
            axis=0,
        )  # shape: (H, W)

    return bayer_f32 * correction.astype(np.float32)


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


# ── Lens distortion correction ─────────────────────────────────────────────────

@dataclass
class DistortionParams:
    """Radial polynomial distortion parameters for one camera.

    Calibrated by the factory and stored in calibration.lri under
    mc.geometry.distortion.polynomial for each module calibration block.

    The forward distortion model (normalised space):
        r²  = xn² + yn²,  where xn = (x − cx) / norm_x
        factor = 1 + k1·r² + k2·r⁴ + k3·r⁶ + k4·r⁸ + k5·r¹⁰
        x_distorted = xn·factor·norm_x + cx

    Undistortion uses this same forward map applied to the *destination* grid
    (inverse mapping), so no iterative solver is required.
    """
    cx: float          # distortion centre x (pixels)
    cy: float          # distortion centre y (pixels)
    norm_x: float      # normalisation scale x (pixels)
    norm_y: float      # normalisation scale y (pixels)
    coeffs: tuple[float, ...]  # (k1, k2, k3, k4, k5) radial polynomial coefficients
    valid_roi: tuple[int, int, int, int] | None  # (x, y, w, h) bounding box, or None


def load_distortion_params(calib_dir: Path, camera_id: CameraId) -> DistortionParams | None:
    """Load radial distortion parameters for one camera from calibration.lri.

    Parses ``<calib_dir>/calibration.lri`` as an LELR block stream, finds the
    FactoryModuleCalibration entry for the requested camera, and returns its
    polynomial distortion model.

    Accesses proto message fields directly (not via MessageToDict) to preserve
    full float precision in the polynomial coefficients.

    Returns None when the file is absent, unreadable, or contains no distortion
    data for this camera — callers should skip undistortion in that case.
    """
    calib_path = calib_dir / "calibration.lri"
    if not calib_path.exists():
        return None

    from mizukage._block import iter_blocks, BlockType
    import mizukage._proto as _proto

    try:
        data = calib_path.read_bytes()
    except OSError:
        return None

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

        for mc in lh.module_calibration:
            if int(mc.camera_id) != int(camera_id):
                continue
            # Guard: geometry, distortion, and polynomial sub-messages are optional.
            if not mc.HasField("geometry"):
                continue
            if not mc.geometry.HasField("distortion"):
                continue
            if not mc.geometry.distortion.HasField("polynomial"):
                continue

            poly = mc.geometry.distortion.polynomial

            # Extract valid_roi bounding box if present.
            valid_roi: tuple[int, int, int, int] | None = None
            if poly.HasField("valid_roi"):
                roi = poly.valid_roi
                valid_roi = (int(roi.x), int(roi.y), int(roi.width), int(roi.height))

            return DistortionParams(
                cx=float(poly.distortion_center.x),
                cy=float(poly.distortion_center.y),
                norm_x=float(poly.normalization.x),
                norm_y=float(poly.normalization.y),
                coeffs=tuple(float(k) for k in poly.coeffs),
                valid_roi=valid_roi,
            )

    return None


def undistort_image(
    image: np.ndarray,
    params: DistortionParams,
) -> np.ndarray:
    """Undistort a float32 or uint8 image using the factory radial polynomial.

    Uses inverse mapping: for each output pixel, compute where it originated in
    the distorted (input) image using the forward polynomial, then
    bilinear-interpolate the source at that fractional coordinate.

    This avoids iterative inversion — the forward map is applied to the
    *destination* grid, which gives exact inverse-map coordinates directly.

    image: (H, W) or (H, W, C), any dtype
    Returns: same shape and dtype as input; out-of-bounds border pixels are 0.
    """
    from scipy.ndimage import map_coordinates

    H, W = image.shape[:2]
    in_dtype = image.dtype

    # Build a dense output-pixel grid (rows = y axis, cols = x axis).
    rows, cols = np.mgrid[0:H, 0:W]  # both (H, W) int arrays

    # Normalise to the distortion model's coordinate space.
    xn = (cols.astype(np.float64) - params.cx) / params.norm_x
    yn = (rows.astype(np.float64) - params.cy) / params.norm_y
    r2 = xn ** 2 + yn ** 2

    # Evaluate the radial distortion polynomial:
    #   factor = 1 + k1·r² + k2·r⁴ + k3·r⁶ + ...
    factor = np.ones_like(r2)
    r2k = r2.copy()
    for k in params.coeffs:
        factor += k * r2k
        r2k *= r2

    # Source pixel coordinates (where to sample from the distorted input).
    # map_coordinates uses (row, col) order, so (src_y, src_x).
    src_x = xn * factor * params.norm_x + params.cx   # float64 (H, W)
    src_y = yn * factor * params.norm_y + params.cy   # float64 (H, W)
    coords = np.array([src_y, src_x])                 # shape (2, H, W)

    # Bilinear interpolation (order=1); pixels outside the source image → 0.
    if image.ndim == 2:
        # Grayscale / single-channel image
        result = map_coordinates(
            image.astype(np.float64), coords,
            order=1, mode="constant", cval=0.0,
        )
        return result.astype(in_dtype)
    else:
        # Multi-channel image (H, W, C): process each channel independently.
        channels = image.shape[2]
        img_f64 = image.astype(np.float64)
        out_f64 = np.empty((H, W, channels), dtype=np.float64)
        for c in range(channels):
            out_f64[..., c] = map_coordinates(
                img_f64[..., c], coords,
                order=1, mode="constant", cval=0.0,
            )
        return out_f64.astype(in_dtype)
