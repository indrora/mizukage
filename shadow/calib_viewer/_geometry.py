"""Geometric intrinsics / distortion panel for shadow calib-view."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from shadow.calib_viewer._data import CalibData

_TAG_GROUP = "geo_group"


def build(data: "CalibData", init_camera: str | None) -> None:
    """Build the geometry tab skeleton."""
    import dearpygui.dearpygui as dpg

    dpg.add_text("Per-focus-bundle intrinsics, extrinsics, and distortion coefficients.")
    dpg.add_separator()
    with dpg.group(tag=_TAG_GROUP):
        pass

    if init_camera:
        update(data, init_camera)


def update(data: "CalibData", camera: str) -> None:
    """Rebuild geometry panel for the selected camera."""
    import dearpygui.dearpygui as dpg

    dpg.delete_item(_TAG_GROUP, children_only=True)

    geo_list = data.geometry.get(camera)
    if not geo_list:
        dpg.add_text(f"No geometry data for {camera}.", parent=_TAG_GROUP)
        return

    geo = geo_list[0]  # first (primary) geometry block
    mirror = geo.get("mirror_type", "NONE")
    mirror_label = {"NONE": "Fixed", "GLUED": "Glued", "MOVABLE": "Movable"}.get(mirror, mirror)

    with dpg.group(parent=_TAG_GROUP):
        dpg.add_text(f"Mirror type: {mirror_label}")

        # Hall-code range
        lhcr = geo.get("lens_hall_code_range", {})
        if lhcr:
            dpg.add_text(f"Lens hall-code range: {lhcr.get('min', '?')} – {lhcr.get('max', '?')}")

        fdr = geo.get("focus_distance_range", {})
        if fdr:
            dpg.add_text(
                f"Focus distance range: {fdr.get('min', '?'):.0f} – {fdr.get('max', '?'):.0f} mm"
                if isinstance(fdr.get("min"), (int, float)) else ""
            )

        dpg.add_separator()

        # Per-focus-bundle intrinsics table
        bundles: list[dict] = geo.get("per_focus_calibration", [])
        if not bundles:
            dpg.add_text("No focus bundle data.")
            return

        dpg.add_text(f"Focus bundles: {len(bundles)}", color=[200, 200, 100])

        with dpg.table(
            header_row=True,
            borders_innerH=True,
            borders_innerV=True,
            borders_outerH=True,
            borders_outerV=True,
            resizable=True,
            width=-1,
        ):
            for label in ["Hall", "Dist (mm)", "fx", "fy", "cx", "cy", "RMS", "Reproj"]:
                dpg.add_table_column(label=label)

            for b in bundles:
                hall = b.get("focus_hall_code", "—")
                dist = b.get("focus_distance", "—")
                intr = b.get("intrinsics", {})
                km   = intr.get("k_mat", {})
                rms  = intr.get("rms_error", None)
                extr = b.get("extrinsics", {})
                can  = extr.get("canonical", {})
                rp   = can.get("reprojection_error", None)

                fx  = km.get("x00", float("nan"))
                fy  = km.get("x11", float("nan"))
                cx  = km.get("x02", float("nan"))
                cy  = km.get("x12", float("nan"))

                def _f(v: Any) -> str:
                    if isinstance(v, float) and v != v:
                        return "—"
                    if isinstance(v, float):
                        return f"{v:.1f}"
                    return str(v)

                with dpg.table_row():
                    dpg.add_text(_f(hall))
                    dpg.add_text(_f(dist) if not isinstance(dist, float) else f"{dist:.0f}")
                    dpg.add_text(_f(fx))
                    dpg.add_text(_f(fy))
                    dpg.add_text(_f(cx))
                    dpg.add_text(_f(cy))
                    dpg.add_text(f"{rms:.4f}" if isinstance(rms, float) else "—")
                    dpg.add_text(f"{rp:.4f}" if isinstance(rp, float) else "—")

        # Distortion coefficients
        dist_d = geo.get("distortion", {})
        if dist_d:
            dpg.add_separator()
            dpg.add_text("Distortion coefficients", color=[200, 200, 100])
            coeffs = {k: v for k, v in dist_d.items() if isinstance(v, (int, float))}
            if coeffs:
                with dpg.table(header_row=True, resizable=True, width=-1):
                    for k in coeffs:
                        dpg.add_table_column(label=k)
                    with dpg.table_row():
                        for v in coeffs.values():
                            dpg.add_text(f"{v:.6f}")
            else:
                # Distortion might be a nested message
                dpg.add_text(str(dist_d))
