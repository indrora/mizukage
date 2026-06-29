"""shadow export — save camera module images as PNG or TIFF."""
from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

import shadow
from shadow._types import CameraId

console = Console()


@click.command("export")
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
@click.argument("out_dir", default=".", type=click.Path(file_okay=False))
@click.option(
    "--camera", "-c",
    multiple=True,
    metavar="CAMERA",
    help="Camera(s) to export (e.g. A1, B3). Repeatable. Default: all.",
)
@click.option(
    "--format", "-f",
    "fmt",
    type=click.Choice(["png", "tiff"], case_sensitive=False),
    default="png",
    show_default=True,
    help="Output image format.",
)
@click.option(
    "--raw",
    is_flag=True,
    default=False,
    help="Export raw 16-bit grayscale Bayer (no debayering).",
)
@click.option(
    "--half-res",
    is_flag=True,
    default=False,
    help="Use fast half-resolution debayer (ignored with --raw).",
)
@click.option(
    "--no-subtract-black",
    is_flag=True,
    default=False,
    help="Skip black level subtraction.",
)
@click.option(
    "--no-awb",
    is_flag=True,
    default=False,
    help="Skip white-balance gains (export raw linear colour).",
)
def export(
    file: str,
    out_dir: str,
    camera: tuple[str, ...],
    fmt: str,
    raw: bool,
    half_res: bool,
    no_subtract_black: bool,
    no_awb: bool,
) -> None:
    """Export camera module images from an LRI file.

    Saves one image per camera module to OUT_DIR (default: current directory).
    Default output is 8-bit RGB PNG (debayered). Use --raw for 16-bit
    grayscale Bayer, or --format tiff for TIFF output.

    PNG export:   A1.png, B3.png, ...   (8-bit RGB) or A1_raw.png (16-bit gray)
    TIFF export:  A1.tiff, B3.tiff, ... (same conventions)

    Examples:

        shadow export photo.lri ./out
        shadow export photo.lri ./out --raw --format tiff
        shadow export photo.lri ./out --camera B4 --half-res
    """
    lri = shadow.open_lri(file)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    images = _filter_cameras(lri.images, camera)
    if not images:
        console.print("[yellow]No matching camera images found.[/yellow]")
        sys.exit(1)

    subtract_black = not no_subtract_black
    apply_awb = not no_awb
    ext = "." + fmt.lower()
    suffix = "_raw" if raw else ""

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Exporting...", total=len(images))

        for img in images:
            name = img.camera_id.name
            dest = out / f"{name}{suffix}{ext}"
            progress.update(task, description=f"Exporting {name}")

            if fmt.lower() == "tiff":
                img.to_tiff(dest, raw=raw, half_res=half_res,
                            subtract_black=subtract_black, apply_awb=apply_awb)
            else:
                img.to_png(dest, raw=raw, half_res=half_res,
                           subtract_black=subtract_black, apply_awb=apply_awb)

            progress.advance(task)

    ref_name = lri.metadata.reference_camera.name if lri.metadata.reference_camera else "—"
    console.print(
        f"[green]Exported {len(images)} image(s) to[/green] {out}"
        f"  [dim](reference: {ref_name})[/dim]"
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
