"""shadow.calib_viewer — DearPyGui calibration data explorer.

Entry point: run_viewer(calib_dir)
Requires: pip install 'shadow[explorer]'
"""
from __future__ import annotations

from pathlib import Path

# Window geometry defaults
_WIN_W = 1280
_WIN_H = 840
_SIDEBAR_W = 170


def run_viewer(calib_dir: Path) -> None:
    """Open the interactive calibration viewer for *calib_dir*."""
    import dearpygui.dearpygui as dpg

    from shadow.calib_viewer._data import load_calib_data
    from shadow.calib_viewer import _hotpixels, _noise, _vignetting, _geometry, _color, _layout

    # ── Load calibration data (once, at startup) ─────────────────────────────
    data = load_calib_data(calib_dir)

    init_cam: str | None = data.cameras[0] if data.cameras else None

    # ── DearPyGui setup ───────────────────────────────────────────────────────
    dpg.create_context()
    dpg.create_viewport(
        title=f"shadow calib-view — {calib_dir}",
        width=_WIN_W,
        height=_WIN_H,
        min_width=800,
        min_height=500,
    )

    with dpg.window(tag="main_window"):
        with dpg.group(horizontal=True):
            # ── Left sidebar ──────────────────────────────────────────────────
            with dpg.child_window(
                width=_SIDEBAR_W,
                border=True,
                tag="sidebar",
            ):
                dpg.add_text("Cameras", color=[160, 200, 230])
                dpg.add_separator()

                if data.cameras:
                    dpg.add_radio_button(
                        items=data.cameras,
                        default_value=init_cam,
                        callback=_on_camera_select,
                        user_data={"data": data, "panels": None},  # panels filled below
                        tag="camera_radio",
                        indent=4,
                    )
                else:
                    dpg.add_text("(no cameras)", color=[180, 100, 100])

                dpg.add_spacer(height=12)
                dpg.add_separator()
                dpg.add_text("Sensor", color=[160, 200, 230])
                dpg.add_text(
                    f"Black: {data.black_level:.0f}"
                    f"\nWhite: {data.white_level:.0f}",
                    tag="sensor_info",
                )
                if data.device_model:
                    dpg.add_spacer(height=6)
                    dpg.add_text(data.device_model, wrap=_SIDEBAR_W - 12, color=[140, 140, 140])
                if data.calib_timestamp:
                    dpg.add_spacer(height=6)
                    dpg.add_text("Calibrated:", color=[160, 200, 230])
                    dpg.add_text(data.calib_timestamp, wrap=_SIDEBAR_W - 12, color=[140, 140, 140])

            # ── Main content area ─────────────────────────────────────────────
            with dpg.child_window(border=False, tag="content_area"):
                with dpg.tab_bar(tag="tab_bar"):

                    with dpg.tab(label="Hot pixels", tag="tab_hp"):
                        _hotpixels.build(data, init_cam)

                    with dpg.tab(label="Noise model", tag="tab_noise"):
                        _noise.build(data)

                    with dpg.tab(label="Vignetting", tag="tab_vig"):
                        _vignetting.build(data, init_cam)

                    with dpg.tab(label="Geometry", tag="tab_geo"):
                        _geometry.build(data, init_cam)

                    with dpg.tab(label="Color", tag="tab_color"):
                        _color.build(data, init_cam)

                    with dpg.tab(label="Layout", tag="tab_layout"):
                        _layout.build(data)

    # Store panel update references in user_data of the radio button
    dpg.set_item_user_data(
        "camera_radio",
        {
            "data": data,
            "panels": {
                "hotpixels":  _hotpixels.update,
                "vignetting": _vignetting.update,
                "geometry":   _geometry.update,
                "color":      _color.update,
                # noise model has global sensor data — no per-camera update needed
            },
        },
    )

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main_window", True)
    dpg.start_dearpygui()
    dpg.destroy_context()


def _on_camera_select(sender: str | int, app_data: str, user_data: dict) -> None:
    """Called when the user clicks a camera in the sidebar radio list."""
    camera = app_data
    data   = user_data["data"]
    panels = user_data.get("panels") or {}

    for update_fn in panels.values():
        try:
            update_fn(data, camera)
        except Exception as exc:
            # Never crash the UI loop on a panel update error.
            import traceback
            traceback.print_exc()
