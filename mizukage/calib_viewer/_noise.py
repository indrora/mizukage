"""VST noise model panel for shadow calib-view."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mizukage.calib_viewer._data import CalibData

_TAG_PLOT  = "noise_plot"
_TAG_Y     = "noise_y"
_TAG_R     = "noise_r"
_TAG_G     = "noise_g"
_TAG_B     = "noise_b"
_TAG_DRAG  = "noise_gain_drag"
_TAG_LABEL = "noise_info"

# Line colors for R/G/B series
_COL_R = [220, 80, 80, 255]
_COL_G = [80, 200, 80, 255]
_COL_B = [80, 120, 220, 255]


def build(data: "CalibData") -> None:
    """Build the VST noise model tab content under the current DPG parent."""
    import dearpygui.dearpygui as dpg

    entries = data.vst_entries
    wl = data.white_level if data.white_level > 0 else 1023.0
    mid = wl / 2.0

    if not entries:
        dpg.add_text("No VST noise model found in calibration.lri.")
        return

    gains  = [float(e.gain_x100) for e in entries]
    r_sig  = [_sigma(e.r_a, e.r_b, mid, wl) for e in entries]
    g_sig  = [_sigma(e.g_a, e.g_b, mid, wl) for e in entries]
    b_sig  = [_sigma(e.b_a, e.b_b, mid, wl) for e in entries]
    init_x = gains[len(gains) // 2]

    dpg.add_text(
        "Sigma at mid-signal versus analog gain (green channel most relevant for luma-weighted denoisers).",
        wrap=0,
    )

    with dpg.plot(
        tag=_TAG_PLOT,
        height=420,
        width=-1,
        no_title=True,
    ):
        dpg.add_plot_legend()
        dpg.add_plot_axis(dpg.mvXAxis, label="Gain code (×100)")
        with dpg.plot_axis(dpg.mvYAxis, label="Sigma (normalised [0,1])", tag=_TAG_Y):
            r_series = dpg.add_line_series(gains, r_sig, label="R", tag=_TAG_R)
            g_series = dpg.add_line_series(gains, g_sig, label="G", tag=_TAG_G)
            b_series = dpg.add_line_series(gains, b_sig, label="B", tag=_TAG_B)

        dpg.add_drag_line(
            tag=_TAG_DRAG,
            label="Gain",
            color=[255, 220, 0, 200],
            default_value=init_x,
            callback=_on_drag,
            vertical=True,
        )

    # Apply per-series colours via item themes
    for series_tag, col in ((r_series, _COL_R), (g_series, _COL_G), (b_series, _COL_B)):
        with dpg.theme() as t:
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, col, category=dpg.mvThemeCat_Plots)
        dpg.bind_item_theme(series_tag, t)

    dpg.add_text("", tag=_TAG_LABEL)
    _update_label(entries, wl, init_x)


def _sigma(a: float, b: float, mid: float, wl: float) -> float:
    v = a * mid + b
    return (max(0.0, v) ** 0.5) / wl


def _on_drag(sender: str | int, app_data: float, user_data) -> None:
    """Recompute sigma label whenever the drag line moves."""
    import dearpygui.dearpygui as dpg
    try:
        val = float(dpg.get_value(_TAG_DRAG))
        xy_r = dpg.get_value(_TAG_R)
        xy_g = dpg.get_value(_TAG_G)
        xy_b = dpg.get_value(_TAG_B)
        x_data = xy_r[0]
        if not x_data:
            return
        idx = min(range(len(x_data)), key=lambda i: abs(x_data[i] - val))
        gain = x_data[idx]
        dpg.set_value(
            _TAG_LABEL,
            f"Gain {gain/100:1.2f}×  →  "
            f"σ_R={xy_r[1][idx]:.4g}  "
            f"σ_G={xy_g[1][idx]:.4g}  "
            f"σ_B={xy_b[1][idx]:.4g}",
        )
    except Exception:
        pass


def _update_label(entries: list, wl: float, x: float) -> None:
    """Set initial sigma label for the entry nearest *x*."""
    import dearpygui.dearpygui as dpg
    if not entries:
        return
    e = min(entries, key=lambda e: abs(e.gain_x100 - x))
    mid = wl / 2.0
    try:
        dpg.set_value(
            _TAG_LABEL,
            f"Gain {e.gain_x100/100:1.2f}×  →  "
            f"σ_R={_sigma(e.r_a,e.r_b,mid,wl):.4g}  "
            f"σ_G={_sigma(e.g_a,e.g_b,mid,wl):.4g}  "
            f"σ_B={_sigma(e.b_a,e.b_b,mid,wl):.4g}",
        )
    except Exception:
        pass
