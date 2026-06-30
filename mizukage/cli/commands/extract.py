"""shadow extract — save raw Bayer data as .npy arrays."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

import mizukage
from mizukage._types import CameraId

console = Console()


@click.command("extract")
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
@click.argument("out_dir", default=".", type=click.Path(file_okay=False))
@click.option(
    "--camera", "-c",
    multiple=True,
    metavar="CAMERA",
    help="Camera(s) to extract (e.g. A1, B3). Repeatable. Default: all.",
)
@click.option(
    "--no-subtract-black",
    is_flag=True,
    default=False,
    help="Skip black level subtraction.",
)
@click.option(
    "--no-metadata",
    is_flag=True,
    default=False,
    help="Skip writing metadata.json.",
)
def extract(
    file: str,
    out_dir: str,
    camera: tuple[str, ...],
    no_subtract_black: bool,
    no_metadata: bool,
) -> None:
    """Extract raw Bayer pixel data from an LRI file as .npy arrays.

    Saves one uint16 numpy array per camera module to OUT_DIR
    (default: current directory). Also writes metadata.json unless
    --no-metadata is given.

    Example:

        shadow extract photo.lri ./raw_arrays --camera B4 --camera C1
    """
    lri = shadow.open_lri(file)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Resolve requested cameras
    images = _filter_cameras(lri.images, camera)
    if not images:
        console.print("[yellow]No matching camera images found.[/yellow]")
        sys.exit(1)

    subtract_black = not no_subtract_black

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Extracting...", total=len(images))

        for img in images:
            name = img.camera_id.name
            progress.update(task, description=f"Extracting {name}")
            arr = img.to_raw_numpy(subtract_black=subtract_black)
            np.save(out / f"{name}.npy", arr)
            progress.advance(task)

    if not no_metadata:
        _write_metadata_json(lri, out)

    console.print(
        f"[green]Extracted {len(images)} module(s) to[/green] {out}"
    )


def _filter_cameras(images, camera_names: tuple[str, ...]):
    if not camera_names:
        return list(images)
    requested: set[CameraId] = set()
    for name in camera_names:
        try:
            requested.add(CameraId.from_name(name))
        except KeyError:
            console.print(f"[red]Unknown camera: {name!r}[/red]")
            sys.exit(1)
    return [img for img in images if img.camera_id in requested]


def _write_metadata_json(lri: shadow.LriFile, out: Path) -> None:
    meta = lri.metadata

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
        "cameras": {
            img.camera_id.name: {
                "sensor": img.sensor_model.name,
                "width": img.width,
                "height": img.height,
                "format": img.raw_format.name,
                "cfa": img.cfa_pattern.name if img.cfa_pattern else None,
                "exposure_ms": img.exposure_ms,
                "analog_gain": img.analog_gain,
                "digital_gain": img.digital_gain,
                "flip_h": img.flip_h,
                "flip_v": img.flip_v,
                "black_level": img._black_level,
            }
            for img in lri.images
        },
    }

    path = out / "metadata.json"
    path.write_text(json.dumps(data, indent=2))
    console.print(f"  Metadata: {path.name}")
