"""shadow calib-view — interactive calibration data viewer."""
from __future__ import annotations

from pathlib import Path

import click


@click.command("calib-view")
@click.argument("calib_dir", type=click.Path(exists=True, file_okay=False))
def calib_view(calib_dir: str) -> None:
    """Interactively visualize calibration data from a lightcal directory.

    Opens a DearPyGui window. Select a camera from the left sidebar to
    update all per-camera tabs. Sensor black/white levels, device model,
    and calibration date are shown in the sidebar below the camera list.

    Tabs:

    \b
      Hot pixels  — sqrt-scaled defect-density heatmap (104x78 grid);
                    per-measurement gain, temperature, and exposure time.
      Noise model — VST sigma-vs-gain curves for R, Gr, Gb, B channels.
      Vignetting  — falloff correction grid; hall-code selector for
                    C-array cameras with movable mirrors.
      Geometry    — intrinsics table per focus bundle (fx, fy, cx, cy,
                    RMS error, sensor temperature); radial distortion
                    coefficients (k1-k5, centre, normalisation); ideal-
                    vs-distorted grid visualisation; rotation matrix and
                    camera world position from extrinsics.
      Color       — factory forward and colour matrices per illuminant;
                    neutral-point locus scatter plot (rg/bg ratios).
      Layout      — bird's-eye position map for all 16 modules, grouped
                    by focal-length array (A/B/C).

    Requires: pip install 'shadow[explorer]'
    """
    try:
        import dearpygui.dearpygui as dpg  # noqa: F401
    except ImportError:
        raise click.UsageError(
            "shadow[explorer] is not installed.\n"
            "Run: pip install 'shadow[explorer]'"
        )
    from mizukage.calib_viewer import run_viewer
    run_viewer(Path(calib_dir))
