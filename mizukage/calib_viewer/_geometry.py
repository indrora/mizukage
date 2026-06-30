"""Geometric intrinsics / distortion panel for shadow calib-view."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from shadow.calib_viewer._data import CalibData

_TAG_GROUP = "geo_group"

# Grid density for distortion visualisation
_VIZ_NX = 17
_VIZ_NY = 13

# L16 sensor dimensions (physical pixel array)
_SENSOR_W = 4160
_SENSOR_H = 3120

# Canvas scale: 1:8 reduction → 520×390 px canvas
_VIZ_SCALE = 1 / 8
_VIZ_CW    = int(_SENSOR_W * _VIZ_SCALE)   # 520
_VIZ_CH    = int(_SENSOR_H * _VIZ_SCALE)   # 390


def _distortion_viz(dist_d: dict) -> None:
    """Draw ideal vs. radially-distorted grid to visualise the lens distortion model."""
    import dearpygui.dearpygui as dpg

    poly = dist_d.get("polynomial", {})
    if not poly:
        return
    center = poly.get("distortion_center", {})
    norm_p = poly.get("normalization", {})
    k_list = [float(c) for c in poly.get("coeffs", [])]
    if not k_list:
        return

    cx     = float(center.get("x", _SENSOR_W / 2))
    cy     = float(center.get("y", _SENSOR_H / 2))
    norm_x = float(norm_p.get("x", float(_SENSOR_W)))
    norm_y = float(norm_p.get("y", norm_x))

    # ── Apply forward distortion model to a regular grid ─────────────────────
    # xn = (x - cx) / norm_x;  factor = 1 + k1·r² + k2·r⁴ + …
    xs = np.linspace(0.0, _SENSOR_W, _VIZ_NX)
    ys = np.linspace(0.0, _SENSOR_H, _VIZ_NY)
    gx, gy = np.meshgrid(xs, ys)           # (NY, NX)

    xn = (gx - cx) / norm_x
    yn = (gy - cy) / norm_y
    r2 = xn ** 2 + yn ** 2

    factor = np.ones_like(r2)
    r2k = r2.copy()
    for k in k_list:
        factor += k * r2k
        r2k *= r2

    dx = xn * factor * norm_x + cx         # distorted x in sensor px
    dy = yn * factor * norm_y + cy         # distorted y in sensor px

    # Max corner displacement (for the info label)
    ideal_corners = np.array([
        [gx[0, 0], gy[0, 0]], [gx[0, -1], gy[0, -1]],
        [gx[-1, 0], gy[-1, 0]], [gx[-1, -1], gy[-1, -1]],
    ])
    dist_corners = np.array([
        [dx[0, 0], dy[0, 0]], [dx[0, -1], dy[0, -1]],
        [dx[-1, 0], dy[-1, 0]], [dx[-1, -1], dy[-1, -1]],
    ])
    max_disp = float(np.max(np.linalg.norm(dist_corners - ideal_corners, axis=1)))
    k1_sign  = "barrel" if k_list[0] < 0 else "pincushion" if k_list[0] > 0 else "none"

    # ── Draw ─────────────────────────────────────────────────────────────────
    # Scale to canvas coordinates
    s = _VIZ_SCALE
    igx = (gx * s).tolist()   # ideal, canvas scale
    igy = (gy * s).tolist()
    wgx = (dx * s).tolist()   # warped
    wgy = (dy * s).tolist()

    IDEAL_COL  = [55, 60, 80, 200]
    WARP_COL   = [220, 200, 80, 255]
    CENTER_COL = [220, 80, 80, 255]

    dpg.add_text(
        f"Distortion grid  ({k1_sign}, k1={k_list[0]:+.4f})"
        f"  max corner shift: {max_disp:.1f} px",
        color=[160, 160, 160],
    )
    dpg.add_text("  grey = ideal  amber = distorted", color=[120, 120, 120])

    with dpg.drawlist(width=_VIZ_CW, height=_VIZ_CH):
        dpg.draw_rectangle([0, 0], [_VIZ_CW - 1, _VIZ_CH - 1],
                           fill=[15, 16, 22, 255], color=[40, 40, 50, 255])

        # Ideal grid — horizontal then vertical segments
        for j in range(_VIZ_NY):
            for i in range(_VIZ_NX - 1):
                dpg.draw_line([igx[j][i], igy[j][i]], [igx[j][i+1], igy[j][i+1]],
                              color=IDEAL_COL, thickness=1)
        for j in range(_VIZ_NY - 1):
            for i in range(_VIZ_NX):
                dpg.draw_line([igx[j][i], igy[j][i]], [igx[j+1][i], igy[j+1][i]],
                              color=IDEAL_COL, thickness=1)

        # Distorted grid
        for j in range(_VIZ_NY):
            for i in range(_VIZ_NX - 1):
                dpg.draw_line([wgx[j][i], wgy[j][i]], [wgx[j][i+1], wgy[j][i+1]],
                              color=WARP_COL, thickness=1)
        for j in range(_VIZ_NY - 1):
            for i in range(_VIZ_NX):
                dpg.draw_line([wgx[j][i], wgy[j][i]], [wgx[j+1][i], wgy[j+1][i]],
                              color=WARP_COL, thickness=1)

        # Distortion centre crosshair
        ccx, ccy = cx * s, cy * s
        r = 5
        dpg.draw_circle([ccx, ccy], r, color=CENTER_COL, thickness=1)
        dpg.draw_line([ccx - r * 2, ccy], [ccx + r * 2, ccy], color=CENTER_COL, thickness=1)
        dpg.draw_line([ccx, ccy - r * 2], [ccx, ccy + r * 2], color=CENTER_COL, thickness=1)


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
            for label in ["Hall", "Dist (mm)", "fx", "fy", "cx", "cy", "RMS", "Reproj", "Temp (C)"]:
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
                temp = b.get("sensor_temp", None)

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
                    dpg.add_text(f"{temp:.0f}" if isinstance(temp, (int, float)) else "—")

        # Distortion coefficients + grid visualisation
        dist_d = geo.get("distortion", {})
        if dist_d:
            dpg.add_separator()
            dpg.add_text("Distortion model", color=[200, 200, 100])
            poly = dist_d.get("polynomial", {})
            if poly:
                center = poly.get("distortion_center", {})
                norm_p = poly.get("normalization", {})
                k_list = [float(c) for c in poly.get("coeffs", [])]
                cx_d   = float(center.get("x", 0.0))
                cy_d   = float(center.get("y", 0.0))
                nx_d   = float(norm_p.get("x", 0.0))
                dpg.add_text(
                    f"Centre: ({cx_d:.1f}, {cy_d:.1f} px)   Norm: {nx_d:.0f} px"
                )
                if k_list:
                    with dpg.table(header_row=True, resizable=False):
                        for i in range(len(k_list)):
                            dpg.add_table_column(
                                label=f"k{i+1}", width_fixed=True, init_width_or_weight=100
                            )
                        with dpg.table_row():
                            for v in k_list:
                                dpg.add_text(f"{v:+.6f}")
            else:
                # Flat scalar fields (legacy format)
                flat = {k: v for k, v in dist_d.items() if isinstance(v, (int, float))}
                if flat:
                    with dpg.table(header_row=True, resizable=True, width=-1):
                        for k in flat:
                            dpg.add_table_column(label=k)
                        with dpg.table_row():
                            for v in flat.values():
                                dpg.add_text(f"{v:.6f}")
                else:
                    dpg.add_text(str(dist_d))
            dpg.add_spacer(height=6)
            _distortion_viz(dist_d)

        # ── Extrinsics ────────────────────────────────────────────────────────
        ext = data.extrinsics.get(camera)
        if ext:
            dpg.add_separator()
            dpg.add_text("Extrinsics", color=[200, 200, 100])
            dpg.add_text(f"Mirror: {ext['mirror_type']}")

            loc = ext.get("camera_loc")
            if loc:
                dpg.add_text(
                    f"Camera world position: [{loc[0]:.2f}, {loc[1]:.2f}, {loc[2]:.2f}] mm"
                )

            R = ext.get("R")
            if R is not None:
                tvec = ext["t"]
                dpg.add_text(
                    f"Translation: [{tvec[0]:.3f}, {tvec[1]:.3f}, {tvec[2]:.3f}] mm"
                )
                dpg.add_text("Rotation matrix (world→cam):", color=[160, 160, 160])
                # 3×3 table; cells colour-coded: bright green near +1, red near -1, grey near 0
                with dpg.table(
                    header_row=False,
                    borders_innerH=True,
                    borders_innerV=True,
                    borders_outerH=True,
                    borders_outerV=True,
                ):
                    for _ in range(3):
                        dpg.add_table_column()
                    for row in R:
                        with dpg.table_row():
                            for v in row:
                                abs_v = abs(v)
                                if abs_v > 0.5:
                                    col = [80, 200, 80, 255] if v > 0 else [220, 80, 80, 255]
                                else:
                                    col = [140, 140, 140, 255]
                                dpg.add_text(f"{v:+.4f}", color=col)

            mi = ext.get("mirror_info")
            if mi:
                ax = mi["rotation_axis"]
                dpg.add_text(
                    f"Rotation axis: [{ax[0]:.4f}, {ax[1]:.4f}, {ax[2]:.4f}]"
                )
                dpg.add_text(
                    f"Mirror angle: offset={mi['mirror_angle_offset']:.2f}°"
                    f"  scale={mi['mirror_angle_scale']:.4f}°/unit"
                )
