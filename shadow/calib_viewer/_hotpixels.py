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
    """Downsample bitmap to display size and encode as flat RGBA floats."""
    from PIL import Image
    H, W = bitmap.shape
    img = Image.fromarray((bitmap * 255).astype(np.uint8), mode="L")
    img = img.resize((_DISP_W, _DISP_H), Image.NEAREST)
    arr = np.array(img, dtype=np.uint8)

    hot = arr > 0
    rgba = np.empty((_DISP_H, _DISP_W, 4), dtype=np.float32)
    rgba[hot]  = [1.0, 0.1, 0.1, 1.0]   # red for hot pixels
    rgba[~hot] = [0.08, 0.08, 0.08, 1.0] # dark grey for normal
    return rgba.flatten().tolist()


def build(data: "CalibData", init_camera: str | None) -> None:
    """Build the hot-pixels tab. Creates the texture registry and image widget."""
    import dearpygui.dearpygui as dpg

    with dpg.texture_registry(show=False):
        dpg.add_raw_texture(
            width=_DISP_W,
            height=_DISP_H,
            default_value=_blank_rgba(),
            format=dpg.mvFormat_Float_rgba,
            tag=_TAG_TEX,
        )

    dpg.add_text("Camera hot-pixel bitmap (red = hot, dark grey = normal). Downsampled for display.")
    dpg.add_image(_TAG_TEX, tag=_TAG_IMG)
    dpg.add_text("—", tag=_TAG_STATS)
    dpg.add_text("", tag=_TAG_MEAS)

    if init_camera:
        update(data, init_camera)


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

    n_hot  = int(bitmap.sum())
    total  = bitmap.size
    pct    = n_hot / total * 100 if total else 0.0
    dpg.set_value(_TAG_STATS, f"{n_hot:,} hot pixels  ({pct:.3f}%)  —  sensor {bitmap.shape[1]}×{bitmap.shape[0]}")

    if stats:
        lines = []
        for i, m in enumerate(stats):
            lines.append(
                f"Measurement {i+1}: gain={m['sensor_gain']:.2f}  "
                f"temp={m['sensor_temperature_c']:.1f}°C  "
                f"exp={m['sensor_exposure_us']} µs"
            )
        dpg.set_value(_TAG_MEAS, "\n".join(lines))
    else:
        dpg.set_value(_TAG_MEAS, "")
