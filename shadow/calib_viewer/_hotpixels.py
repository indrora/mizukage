"""Hot-pixel map panel for shadow calib-view."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from shadow.calib_viewer._data import CalibData

# Fixed display size — all sensors resized to this before upload.
_DISP_W = 416
_DISP_H = 312

_TAG_TEX   = "hotpixel_tex"
_TAG_IMG   = "hotpixel_img"
_TAG_STATS = "hotpixel_stats"
_TAG_MEAS  = "hotpixel_meas"


def _blank_rgba() -> list[float]:
    """Return a flat RGBA float list for the blank/empty texture."""
    return [0.08, 0.08, 0.08, 1.0] * (_DISP_W * _DISP_H)


def _bitmap_to_rgba(bitmap: np.ndarray) -> list[float]:
    """Downsample bitmap to display size and encode as flat RGBA floats.

    The L16 hot-pixel map stores per-pixel severity scores (0=clean, 1–7=defect
    level).  With ~12 % of pixels non-zero, NEAREST downsampling produces a
    confusing speckled pattern.  Instead we use bilinear averaging so each
    display pixel shows the *density* of defects in that region, then colour-map
    that density from dark-grey (0 %) through orange to bright red (≥ 20 %).
    """
    from PIL import Image

    H, W = bitmap.shape
    # Use binary mask (any severity counts) so downsampling gives local defect
    # *fraction* 0.0-1.0 per display pixel.  With ~12% overall density,
    # saturating at 20% (t=1.0) makes the map vividly visible.
    binary = (bitmap > 0).astype(np.uint8) * 255
    img = Image.fromarray(binary, mode="L")
    img = img.resize((_DISP_W, _DISP_H), Image.BILINEAR)
    density = np.array(img, dtype=np.float32) / 255.0  # fraction 0.0-1.0

    # Ramp saturates at 20% density: dark-grey (0%) -> red (>=20%)
    t = np.clip(density * 5.0, 0.0, 1.0)
    rgba = np.empty((_DISP_H, _DISP_W, 4), dtype=np.float32)
    rgba[:, :, 0] = 0.08 + 0.92 * t   # R: dark grey -> bright red
    rgba[:, :, 1] = np.clip(0.08 - 0.07 * t, 0, 1)
    rgba[:, :, 2] = np.clip(0.08 - 0.07 * t, 0, 1)
    rgba[:, :, 3] = 1.0
    return rgba.flatten().tolist()


def register_texture(data: "CalibData", init_camera: str | None) -> None:
    """Create the hot-pixel texture in the app-level texture registry.

    Must be called BEFORE any window is created so DearPyGui registers the
    texture at the top level (not as a child of a tab or window).
    """
    import dearpygui.dearpygui as dpg

    bitmap = data.hot_pixels.get(init_camera) if init_camera else None
    initial = _bitmap_to_rgba(bitmap) if bitmap is not None else _blank_rgba()

    with dpg.texture_registry(show=False):
        dpg.add_raw_texture(
            width=_DISP_W,
            height=_DISP_H,
            default_value=initial,
            format=dpg.mvFormat_Float_rgba,
            tag=_TAG_TEX,
        )


def build(data: "CalibData", init_camera: str | None) -> None:
    """Build the hot-pixels tab widgets (texture must already be registered)."""
    import dearpygui.dearpygui as dpg

    dpg.add_text(
        "Camera hot-pixel bitmap (red = hot, dark grey = normal). Downsampled for display.",
        wrap=0,
    )
    dpg.add_image(_TAG_TEX, tag=_TAG_IMG)

    bitmap = data.hot_pixels.get(init_camera) if init_camera else None
    if bitmap is not None:
        n_defect = int((bitmap > 0).sum())
        total    = bitmap.size
        pct      = n_defect / total * 100
        dpg.add_text(
            f"{n_defect:,} defective pixels  ({pct:.3f}%)  —  sensor {bitmap.shape[1]}×{bitmap.shape[0]}"
            f"  severity 1-{int(bitmap.max())}",
            tag=_TAG_STATS,
        )
    else:
        dpg.add_text("No hot-pixel data for this camera.", tag=_TAG_STATS)

    stats = data.hp_stats.get(init_camera, []) if init_camera else []
    meas_text = _meas_lines(stats)
    dpg.add_text(meas_text, tag=_TAG_MEAS)


def update(data: "CalibData", camera: str) -> None:
    """Refresh the hot-pixel display for the selected camera."""
    import dearpygui.dearpygui as dpg

    bitmap = data.hot_pixels.get(camera)
    stats  = data.hp_stats.get(camera, [])

    if bitmap is None:
        dpg.set_value(_TAG_TEX, _blank_rgba())
        dpg.set_value(_TAG_STATS, "No hot-pixel data for this camera.")
        dpg.set_value(_TAG_MEAS, "")
        return

    dpg.set_value(_TAG_TEX, _bitmap_to_rgba(bitmap))

    n_defect = int((bitmap > 0).sum())
    total    = bitmap.size
    pct      = n_defect / total * 100
    dpg.set_value(
        _TAG_STATS,
        f"{n_defect:,} defective pixels  ({pct:.3f}%)  —  sensor {bitmap.shape[1]}×{bitmap.shape[0]}"
        f"  severity 1-{int(bitmap.max())}",
    )
    dpg.set_value(_TAG_MEAS, _meas_lines(stats))


def _meas_lines(stats: list[dict]) -> str:
    if not stats:
        return ""
    lines = []
    for i, m in enumerate(stats):
        if "sensor_gain" in m:
            lines.append(
                f"Measurement {i+1}: gain={m['sensor_gain']:.2f}  "
                f"temp={m['sensor_temperature_c']:.1f}°C  "
                f"exp={m['sensor_exposure_us']} µs"
            )
    return "\n".join(lines)
