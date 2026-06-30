"""Color matrix panel for shadow calib-view."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shadow.calib_viewer._data import CalibData

_TAG_GROUP  = "color_group"
_TAG_ILLUM  = "color_illum_combo"

# Preferred display order for illuminants
_ILLUM_ORDER = ["D65", "D75", "D50", "A", "F2", "F7", "F11", "TL84", "UNKNOWN"]



def build(data: "CalibData", init_camera: str | None) -> None:
    """Build the color tab skeleton."""
    import dearpygui.dearpygui as dpg

    dpg.add_text("Factory color calibration matrices per illuminant.")
    dpg.add_separator()
    with dpg.group(tag=_TAG_GROUP):
        pass

    if init_camera:
        update(data, init_camera)


def update(data: "CalibData", camera: str) -> None:
    """Rebuild color panel for the selected camera."""
    import dearpygui.dearpygui as dpg

    dpg.delete_item(_TAG_GROUP, children_only=True)

    color_list = data.color.get(camera)
    if not color_list:
        dpg.add_text(f"No color calibration for {camera}.", parent=_TAG_GROUP)
        return

    # Build illuminant → entry mapping
    illum_map: dict[str, dict] = {}
    for entry in color_list:
        t = entry.get("type", "UNKNOWN")
        illum_map.setdefault(t, entry)

    illums = sorted(illum_map.keys(), key=lambda k: _ILLUM_ORDER.index(k) if k in _ILLUM_ORDER else 99)

    _current_illum = [illums[0]]

    def _render(illum: str) -> None:
        dpg.delete_item("color_content", children_only=True)
        entry = illum_map[illum]

        with dpg.group(parent="color_content"):
            rg = entry.get("rg_ratio", None)
            bg = entry.get("bg_ratio", None)
            if rg is not None and bg is not None:
                dpg.add_text(f"Neutral point  rg_ratio={rg:.4f}  bg_ratio={bg:.4f}")

            dpg.add_separator()

            fwd = entry.get("forward_matrix", {})
            ccm = entry.get("color_matrix", {})

            with dpg.group(horizontal=True):
                _mat_widget("Forward matrix (RGB→XYZ)", fwd, "fwd")
                dpg.add_spacer(width=40)
                _mat_widget("Color matrix (XYZ→RGB)", ccm, "ccm")

    with dpg.group(parent=_TAG_GROUP):
        if len(illums) > 1:
            with dpg.group(horizontal=True):
                dpg.add_text("Illuminant:")
                dpg.add_combo(
                    items=illums,
                    default_value=illums[0],
                    width=120,
                    callback=lambda s, a: (_current_illum.__setitem__(0, a), _render(a)),
                )

        with dpg.group(tag="color_content"):
            pass

        # Neutral-point scatter — all illuminants on one plot (static, not per-combo)
        dpg.add_separator()
        dpg.add_text("Neutral points (rg / bg ratios per illuminant)", color=[200, 200, 100])
        _neutral_scatter(illum_map)

    _render(illums[0])


def _neutral_scatter(illum_map: dict[str, dict]) -> None:
    """Render a 2-D scatter of rg_ratio vs bg_ratio for all illuminants."""
    import dearpygui.dearpygui as dpg

    # Gather points
    points: list[tuple[str, float, float]] = []
    for illum, entry in illum_map.items():
        rg = entry.get("rg_ratio")
        bg = entry.get("bg_ratio")
        if rg is not None and bg is not None:
            points.append((illum, float(rg), float(bg)))

    if not points:
        dpg.add_text("No neutral-point data available.", color=[160, 160, 160])
        return

    with dpg.plot(label="Neutral-point locus", height=260, width=380, no_mouse_pos=True):
        dpg.add_plot_axis(dpg.mvXAxis, label="rg ratio")
        with dpg.plot_axis(dpg.mvYAxis, label="bg ratio"):
            for illum, rg, bg in points:
                dpg.add_scatter_series([rg], [bg], label=illum)
        dpg.add_plot_legend()

    # Text list below the plot for precise values
    with dpg.table(
        header_row=True,
        borders_innerV=True,
        borders_outerH=True,
        borders_outerV=True,
        resizable=False,
        width=380,
    ):
        dpg.add_table_column(label="Illuminant", width_fixed=True, init_width_or_weight=90)
        dpg.add_table_column(label="rg ratio",   width_fixed=True, init_width_or_weight=90)
        dpg.add_table_column(label="bg ratio",   width_fixed=True, init_width_or_weight=90)
        for illum, rg, bg in points:
            with dpg.table_row():
                dpg.add_text(illum)
                dpg.add_text(f"{rg:.4f}")
                dpg.add_text(f"{bg:.4f}")


def _mat_widget(title: str, mat_d: dict, tag_prefix: str) -> None:
    """Render a 3×3 matrix as a table with colour-coded cells."""
    import dearpygui.dearpygui as dpg

    vals = [mat_d.get(f"x{r}{c}", 0.0) for r in range(3) for c in range(3)]

    dpg.add_text(title, color=[200, 200, 100])
    with dpg.table(
        header_row=False,
        borders_innerH=True,
        borders_innerV=True,
        borders_outerH=True,
        borders_outerV=True,
        resizable=False,
    ):
        for _ in range(3):
            dpg.add_table_column(width_fixed=True, init_width_or_weight=90)

        for r in range(3):
            with dpg.table_row():
                for c in range(3):
                    v = vals[r * 3 + c]
                    if v > 0.05:
                        color = [100, 220, 100, 255]
                    elif v < -0.05:
                        color = [220, 80, 80, 255]
                    else:
                        color = [170, 170, 170, 255]
                    with dpg.table_cell():
                        dpg.add_text(f"{v:+.4f}", color=color)
