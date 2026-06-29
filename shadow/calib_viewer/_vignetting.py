"""Vignetting heatmap panel for shadow calib-view."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from shadow.calib_viewer._data import CalibData

_TAG_GROUP = "vig_group"


def build(data: "CalibData", init_camera: str | None) -> None:
    """Build the vignetting tab. Content is (re)built per camera selection."""
    import dearpygui.dearpygui as dpg

    dpg.add_text("Vignetting correction grid per camera. Brighter regions require more correction.")
    dpg.add_separator()
    with dpg.group(tag=_TAG_GROUP):
        pass  # placeholder — filled by update()

    if init_camera:
        update(data, init_camera)


def update(data: "CalibData", camera: str) -> None:
    """Rebuild the vignetting panel for the selected camera."""
    import dearpygui.dearpygui as dpg

    dpg.delete_item(_TAG_GROUP, children_only=True)

    vig = data.vignetting.get(camera)
    if not vig:
        dpg.add_text(f"No vignetting data for {camera}.", parent=_TAG_GROUP)
        return

    rel_b: float | None = vig.get("relative_brightness")
    mirror_vigs: list[dict[str, Any]] = vig.get("vignetting", [])

    # Build (label, flat_vals, width, height) tuples
    entries: list[tuple[str, list[float], int, int]] = []
    for mv in mirror_vigs:
        hall = mv.get("hall_code", 0)
        vm   = mv.get("vignetting", {})
        w    = int(vm.get("width",  0))
        h    = int(vm.get("height", 0))
        dp   = vm.get("data_packed", [])
        if w > 0 and h > 0 and dp:
            entries.append((f"Hall {hall}", [float(v) for v in dp], w, h))

    # Crosstalk model fallback (width/height at top level)
    if not entries:
        ct = vig.get("crosstalk", {})
        w  = int(ct.get("width",  0))
        h  = int(ct.get("height", 0))
        dp = ct.get("data_packed", [])
        if w > 0 and h > 0 and dp:
            entries.append(("Crosstalk", [float(v) for v in dp], w, h))

    with dpg.group(parent=_TAG_GROUP):
        if rel_b is not None:
            dpg.add_text(f"Relative brightness: {rel_b:.4f}", color=[200, 200, 100])

        if not entries:
            dpg.add_text("No usable vignetting grid found in calibration data.")
            return

        # State for hall-code switching
        _sel = [0]

        def _render(idx: int) -> None:
            label, vals, w, h = entries[idx]
            lo = min(vals)
            hi = max(vals)

            dpg.delete_item("vig_content", children_only=True)
            with dpg.group(horizontal=True, parent="vig_content"):
                with dpg.plot(
                    label=f"{camera} — {label}",
                    height=350,
                    width=-1,
                    no_mouse_pos=True,
                    no_title=False,
                ) as vig_plot:
                    dpg.add_plot_axis(dpg.mvXAxis, no_gridlines=True)
                    with dpg.plot_axis(dpg.mvYAxis, no_gridlines=True, invert=True):
                        dpg.add_heat_series(
                            vals,
                            rows=h, cols=w,
                            scale_min=lo, scale_max=hi,
                        )
                dpg.bind_colormap(vig_plot, dpg.mvPlotColormap_Viridis)

                dpg.add_colormap_scale(
                    colormap=dpg.mvPlotColormap_Viridis,
                    min_scale=lo,
                    max_scale=hi,
                    height=350,
                    width=80,
                )
            dpg.add_text(
                f"Grid: {w}×{h}  min={lo:.4f}  max={hi:.4f}",
                parent="vig_content",
            )

        if len(entries) > 1:
            labels = [e[0] for e in entries]
            with dpg.group(horizontal=True):
                dpg.add_text("Hall code:")
                dpg.add_combo(
                    items=labels,
                    default_value=labels[0],
                    width=120,
                    callback=lambda s, a: (_sel.__setitem__(0, labels.index(a)), _render(_sel[0])),
                )

        with dpg.group(tag="vig_content"):
            pass

        _render(0)
