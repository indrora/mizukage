"""shadow info — display metadata about an LRI or LRIS file."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

import shadow
from shadow._block import iter_blocks, BlockType, HEADER_SIZE
from shadow._types import CameraId

console = Console()


@click.command("info")
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
@click.option("--blocks", is_flag=True, help="Show raw LELR block breakdown.")
@click.option("--cameras", is_flag=True, help="Show per-camera capture settings.")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def info(file: str, blocks: bool, cameras: bool, as_json: bool) -> None:
    """Display metadata for an LRI or LRIS file."""
    path = Path(file)
    suffix = path.suffix.lower()

    if suffix == ".lris":
        _info_lris(path, as_json)
    else:
        _info_lri(path, blocks, cameras, as_json)


def _info_lri(path: Path, show_blocks: bool, show_cameras: bool, as_json: bool) -> None:
    lri = shadow.open_lri(str(path))
    meta = lri.metadata

    if as_json:
        gps_dict = None
        if meta.gps:
            gps_dict = {
                "latitude": meta.gps.latitude,
                "longitude": meta.gps.longitude,
                "altitude_m": meta.gps.altitude_m,
                "heading": meta.gps.heading,
                "speed": meta.gps.speed,
            }

        data = {
            "file": str(path),
            "size_bytes": path.stat().st_size,
            "image_count": len(lri.images),
            "focal_length_mm": meta.focal_length_mm,
            "reference_camera": meta.reference_camera.name if meta.reference_camera else None,
            "device_model": meta.device_model,
            "firmware_version": meta.firmware_version,
            "hdr_mode": meta.hdr_mode.name if meta.hdr_mode else None,
            "scene_mode": meta.scene_mode.name if meta.scene_mode else None,
            "awb_mode": meta.awb_mode.name if meta.awb_mode else None,
            "awb_gains": {
                "r": meta.awb_gains.r,
                "gr": meta.awb_gains.gr,
                "gb": meta.awb_gains.gb,
                "b": meta.awb_gains.b,
            } if meta.awb_gains else None,
            "on_tripod": meta.on_tripod,
            "gps": gps_dict,
            "cameras": [
                {
                    "camera_id": img.camera_id.name,
                    "sensor": img.sensor_model.name,
                    "width": img.width,
                    "height": img.height,
                    "format": img.raw_format.name,
                    "cfa": img.cfa_pattern.name if img.cfa_pattern else None,
                    "exposure_ms": img.exposure_ms,
                    "analog_gain": img.analog_gain,
                }
                for img in lri.images
            ],
        }
        click.echo(json.dumps(data, indent=2))
        return

    # ── Rich table output ──────────────────────────────────────────────────────
    file_size_mb = path.stat().st_size / (1024 * 1024)

    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="bold cyan", no_wrap=True)
    summary.add_column()

    summary.add_row("File", f"{path.name}  ({file_size_mb:.1f} MB)")
    summary.add_row("Images", str(len(lri.images)))
    summary.add_row("Focal length", f"{meta.focal_length_mm} mm" if meta.focal_length_mm else "n/a")
    if meta.reference_camera is not None:
        summary.add_row("Reference camera", meta.reference_camera.name)
    summary.add_row("Device", meta.device_model or "n/a")
    summary.add_row("Firmware", meta.firmware_version or "n/a")
    if meta.hdr_mode:
        summary.add_row("HDR mode", meta.hdr_mode.name)
    if meta.scene_mode:
        summary.add_row("Scene mode", meta.scene_mode.name)
    if meta.awb_mode:
        summary.add_row("AWB mode", meta.awb_mode.name)
    if meta.awb_gains:
        g = meta.awb_gains
        summary.add_row(
            "AWB gains",
            f"R={g.r:.4f}  Gr={g.gr:.4f}  Gb={g.gb:.4f}  B={g.b:.4f}",
        )
    if meta.on_tripod is not None:
        summary.add_row("On tripod", "yes" if meta.on_tripod else "no")
    if meta.gps:
        gps = meta.gps
        summary.add_row(
            "GPS",
            f"{gps.latitude:.6f}, {gps.longitude:.6f}"
            + (f"  alt {gps.altitude_m:.0f} m" if gps.altitude_m is not None else ""),
        )

    console.print(Panel(summary, title=f"[bold]{path.name}[/bold]", border_style="blue"))

    if show_cameras or not show_blocks:
        _print_camera_table(lri)

    if show_blocks:
        _print_block_table(path)


def _print_camera_table(lri: shadow.LriFile) -> None:
    ref_id = lri.metadata.reference_camera

    tbl = Table(title="Camera modules", box=box.SIMPLE_HEAD, show_lines=False)
    tbl.add_column("Camera", style="bold")
    tbl.add_column("Sensor")
    tbl.add_column("Fmt")
    tbl.add_column("Size", justify="right")
    tbl.add_column("CFA")
    tbl.add_column("Exposure", justify="right")
    tbl.add_column("Gain", justify="right")
    tbl.add_column("Flip", justify="center")

    for img in lri.images:
        is_ref = img.camera_id == ref_id
        cam_label = f"[bold yellow]{img.camera_id.name} *[/bold yellow]" if is_ref else img.camera_id.name
        exposure = f"{img.exposure_ms:.2f} ms"
        gain = f"{img.analog_gain:.3f}"
        cfa = img.cfa_pattern.name if img.cfa_pattern is not None else "mono"
        flip = "".join(
            [("H" if img.flip_h else ""), ("V" if img.flip_v else "")]
        ) or "-"
        tbl.add_row(
            cam_label,
            img.sensor_model.name,
            _short_format(img.raw_format),
            f"{img.width}x{img.height}",
            cfa,
            exposure,
            gain,
            flip,
        )

    console.print(tbl)
    if ref_id is not None:
        console.print("  [yellow]*[/yellow] reference camera\n")


def _short_format(fmt) -> str:
    name = fmt.name
    return name.replace("PACKED_", "").replace("BAYER_", "")


def _print_block_table(path: Path) -> None:
    data = path.read_bytes()

    tbl = Table(title="LELR blocks", box=box.SIMPLE_HEAD)
    tbl.add_column("#", justify="right")
    tbl.add_column("Offset (hex)")
    tbl.add_column("Type")
    tbl.add_column("Block size", justify="right")
    tbl.add_column("Proto offset", justify="right")
    tbl.add_column("Proto size", justify="right")

    for i, (block_start, hdr) in enumerate(iter_blocks(data)):
        size_mb = hdr.block_length / (1024 * 1024)
        size_str = f"{size_mb:.1f} MB" if size_mb >= 1 else f"{hdr.block_length:,} B"
        tbl.add_row(
            str(i),
            f"{block_start:#010x}",
            hdr.msg_type.name,
            size_str,
            str(hdr.msg_offset),
            str(hdr.msg_len),
        )

    console.print(tbl)


def _info_lris(path: Path, as_json: bool) -> None:
    lris = shadow.open_lris(str(path))
    dm = lris.depth_map
    lo, hi = lris.disparity_range

    if as_json:
        click.echo(json.dumps({
            "file": str(path),
            "size_bytes": path.stat().st_size,
            "depth_shape": list(lris.depth_shape),
            "valid_fraction": lris.valid_fraction,
            "disparity_range": [lo, hi],
        }, indent=2))
        return

    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="bold cyan", no_wrap=True)
    summary.add_column()

    file_size_mb = path.stat().st_size / (1024 * 1024)
    summary.add_row("File", f"{path.name}  ({file_size_mb:.1f} MB)")
    summary.add_row("Type", "LRIS sidecar (Lumen depth/calibration)")
    summary.add_row("Depth map shape", f"{dm.shape[0]} rows x {dm.shape[1]} cols")
    summary.add_row("Valid pixels", f"{lris.valid_fraction:.1%}")
    if lo != hi or lo != 0:
        summary.add_row("Disparity range", f"{lo} to {hi}")
    summary.add_row("Depth dtype", str(dm.dtype))

    console.print(Panel(summary, title=f"[bold]{path.name}[/bold]", border_style="green"))
