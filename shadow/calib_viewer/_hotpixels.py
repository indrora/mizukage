"""Hot-pixel map panel for shadow calib-view."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from shadow.calib_viewer._data import CalibData

# Down-sample grid for heat_series display (cols x rows)
_GRID_W = 104   # 4160 / 40
_GRID_H = 78    # 3120 / 40

_TAG_GROUP  = "hp_group"
_TAG_STATS  = "hotpixel_stats"
_TAG_MEAS   = "hotpixel_meas"


def _density_grid(bitmap: np.ndarray) -> np.ndarray:
    """Block-average bitmap to (_GRID_H, _GRID_W) defect-density grid.

    Each cell = fraction of pixels with any non-zero severity in that block.
    Square-root scaled so sparse regions remain visible at ~12% overall density.
    """
    H, W = bitmap.shape
    bh = H // _GRID_H
    bw = W // _GRID_W
    # Trim to exact multiple
    trimmed = (bitmap[:bh * _GRID_H, :bw * _GRID_W] > 0).astype(np.float32)
    density = trimmed.reshape(_GRID_H, bh, _GRID_W, bw).mean(axis=(1, 3))
    # Square-root exaggeration: sqrt(0.12) ≈ 0.35 vs raw 0.12
    return np.sqrt(density)


def build(data: "CalibData", init_camera: str | None) -> None:
    """Build the hot-pixels tab."""
    import dearpygui.dearpygui as dpg

    dpg.add_text(
        "Defective-pixel density map (sqrt-scaled). Bright = high local defect fraction.",
        wrap=0,
    )
    dpg.add_separator()
    with dpg.group(tag=_TAG_GROUP):
        pass
    dpg.add_text("", tag=_TAG_STATS)
    dpg.add_text("", tag=_TAG_MEAS)

    if init_camera:
        update(data, init_camera)


def update(data: "CalibData", camera: str) -> None:
    """Rebuild the heat-series for the selected camera."""
    import dearpygui.dearpygui as dpg

    dpg.delete_item(_TAG_GROUP, children_only=True)

    bitmap = data.hot_pixels.get(camera)
    stats  = data.hp_stats.get(camera, [])

    if bitmap is None:
        dpg.add_text(f"No hot-pixel data for {camera}.", parent=_TAG_GROUP)
        dpg.set_value(_TAG_STATS, "")
        dpg.set_value(_TAG_MEAS, "")
        return

    grid = _density_grid(bitmap)
    lo, hi = float(grid.min()), float(grid.max())

    with dpg.group(horizontal=True, parent=_TAG_GROUP):
        with dpg.plot(
            label=f"Defect density — {camera}",
            height=360,
            width=-80,
            no_mouse_pos=True,
            no_title=False,
        ):
            dpg.add_plot_axis(
                dpg.mvXAxis,
                label=f"col (0-{_GRID_W-1})",
                no_gridlines=True,
                lock_min=True,
                lock_max=True,
            )
            with dpg.plot_axis(
                dpg.mvYAxis,
                label=f"row (0-{_GRID_H-1})",
                no_gridlines=True,
                lock_min=True,
                lock_max=True,
                invert=True,
            ):
                dpg.add_heat_series(
                    grid.flatten().tolist(),
                    rows=_GRID_H,
                    cols=_GRID_W,
                    scale_min=lo,
                    scale_max=hi,
                    bounds_min=(0, 0),
                    bounds_max=(_GRID_W, _GRID_H),
                )
            dpg.bind_colormap(dpg.last_container(), dpg.mvPlotColormap_Plasma)

        dpg.add_colormap_scale(
            colormap=dpg.mvPlotColormap_Plasma,
            min_scale=lo,
            max_scale=hi,
            height=360,
            width=70,
        )

    # Stats
    n_defect = int((bitmap > 0).sum())
    total    = bitmap.size
    pct      = n_defect / total * 100
    dpg.set_value(
        _TAG_STATS,
        f"{n_defect:,} defective pixels  ({pct:.3f}%)  "
        f"sensor {bitmap.shape[1]}x{bitmap.shape[0]}  "
        f"severity 1-{int(bitmap.max())}",
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
                f"temp={m['sensor_temperature_c']:.1f}C  "
                f"exp={m['sensor_exposure_us']} us"
            )
    return "\n".join(lines)
