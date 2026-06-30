"""Camera physical layout tab — top-down scatter of all 16 sensor positions."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mizukage.calib_viewer._data import CalibData

_TAG_PLOT = "layout_plot"

# Focal-length labels and series colours per array letter
_ARRAY_META = {
    "A": {"focal": 28,  "color": [100, 150, 255, 200]},  # blue
    "B": {"focal": 70,  "color": [100, 220, 100, 200]},  # green
    "C": {"focal": 150, "color": [255, 120, 80,  200]},  # red/orange
}

# Canonical display order for the position table
_CAM_ORDER = [
    "A1", "A2", "A3", "A4", "A5",
    "B1", "B2", "B3", "B4", "B5",
    "C1", "C2", "C3", "C4", "C5", "C6",
]


def build(data: "CalibData") -> None:
    """Build the Camera Layout tab.

    Shows a top-down scatter plot of all camera module positions in mm (A1 at
    the origin), grouped by focal-length array, plus a summary table with
    X/Y/Z coordinates and mirror type for every camera.
    """
    import dearpygui.dearpygui as dpg

    dpg.add_text(
        "Physical positions of all 16 camera modules in the sensor plane (mm). "
        "A1 is the coordinate origin. B/C array cameras use real lens positions "
        "from the mirror system; Z depth is shown in the table below.",
        wrap=0,
    )
    dpg.add_separator()

    # ── Group cameras by array letter ─────────────────────────────────────────
    groups: dict[str, list[tuple[str, list[float]]]] = {}
    for cam, ext in data.extrinsics.items():
        loc = ext.get("camera_loc")
        if loc is None:
            continue
        array = cam[0]  # 'A', 'B', or 'C'
        groups.setdefault(array, []).append((cam, loc))

    # ── Scatter plot ──────────────────────────────────────────────────────────
    # equal_aspects keeps X and Y proportional so the layout isn't distorted.
    # It may not be present in all DPG builds, so we try it and fall back.
    def _make_plot(equal_aspects: bool) -> "dpg.plot":  # type: ignore[name-defined]
        kwargs: dict = dict(
            tag=_TAG_PLOT,
            label="Camera positions (top-down view)",
            height=500,
            width=600,
            no_title=True,
        )
        if equal_aspects:
            kwargs["equal_aspects"] = True
        return dpg.plot(**kwargs)

    def _build_series() -> None:
        dpg.add_plot_legend()
        dpg.add_plot_axis(dpg.mvXAxis, label="X (mm)")
        with dpg.plot_axis(dpg.mvYAxis, label="Y (mm)"):
            for array_name in sorted(groups.keys()):
                cams = groups[array_name]
                if not cams:
                    continue
                xs = [c[1][0] for c in cams]
                ys = [c[1][1] for c in cams]
                meta = _ARRAY_META.get(
                    array_name, {"focal": "?", "color": [200, 200, 200, 200]}
                )
                dpg.add_scatter_series(
                    xs, ys,
                    label=f"{array_name}-array ({meta['focal']}mm)",
                )

    try:
        with _make_plot(equal_aspects=True):
            _build_series()
    except TypeError:
        # equal_aspects not supported by this DPG build — retry without it
        with _make_plot(equal_aspects=False):
            _build_series()

    # ── Position table ────────────────────────────────────────────────────────
    dpg.add_separator()
    dpg.add_text("Camera positions:", color=[200, 200, 100])
    with dpg.table(
        header_row=True,
        borders_innerH=True,
        borders_innerV=True,
        borders_outerH=True,
        borders_outerV=True,
        resizable=True,
    ):
        for col_label in ["Camera", "X (mm)", "Y (mm)", "Z (mm)", "Mirror"]:
            dpg.add_table_column(label=col_label)

        for cam in _CAM_ORDER:
            ext = data.extrinsics.get(cam)
            if ext is None:
                continue
            loc = ext.get("camera_loc") or [float("nan")] * 3
            with dpg.table_row():
                dpg.add_text(cam)
                dpg.add_text(f"{loc[0]:.1f}")
                dpg.add_text(f"{loc[1]:.1f}")
                dpg.add_text(f"{loc[2]:.1f}")
                dpg.add_text(ext["mirror_type"])
