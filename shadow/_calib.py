"""shadow._calib — factory calibration helpers.

Provides:
  * VSTEntry / load_vst_model — parse the VST noise model from calibration.lri
  * compute_scalar_sigma      — derive a representative scalar sigma from the model

The VST (Variance-Stabilising Transform) noise model records, for each gain
setting, a per-channel linear model:

    variance(signal) = a * signal + b

where ``signal`` is the raw (pre-black-subtracted) pixel value.  A per-pixel
noise estimate follows:

    sigma(x, y) = sqrt(a * raw_value + b)

This module produces a single scalar sigma representative of the whole image
(green channel at mid-signal), suitable for BM3D, bilateral, and DRUNet.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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

    # Late imports so the module doesn't pull in heavy dependencies at import time.
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
