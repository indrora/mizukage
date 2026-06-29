"""shadow calib-view — interactive calibration data viewer."""
from __future__ import annotations

from pathlib import Path

import click


@click.command("calib-view")
@click.argument("calib_dir", type=click.Path(exists=True, file_okay=False))
def calib_view(calib_dir: str) -> None:
    """Interactively visualize calibration data from a lightcal directory.

    Opens a DearPyGui window showing hot-pixel maps, VST noise curves,
    vignetting heatmaps, geometric intrinsics, and color matrices — one
    pane per camera, switchable from the sidebar.

    Requires: pip install 'shadow[explorer]'
    """
    try:
        import dearpygui.dearpygui as dpg  # noqa: F401
    except ImportError:
        raise click.UsageError(
            "shadow[explorer] is not installed.\n"
            "Run: pip install 'shadow[explorer]'"
        )
    from shadow.calib_viewer import run_viewer
    run_viewer(Path(calib_dir))
