"""shadow export — save camera module images as PNG or TIFF."""
from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

import shadow
from shadow._debayer import DemosaicKernel
from shadow._denoise import DenoiseKernel
from shadow._types import AwbGains, CameraId

console = Console()


class _GammaParamType(click.ParamType):
    """Accepts 'srgb', 'linear', or a positive float."""
    name = "CURVE"

    def convert(self, value, param, ctx):
        if isinstance(value, (bool, float)):
            return value
        low = value.lower()
        if low in ("srgb", "s"):
            return True
        if low in ("linear", "none", "off"):
            return False
        try:
            g = float(value)
        except ValueError:
            self.fail(f"{value!r} is not 'srgb', 'linear', or a number", param, ctx)
        if g <= 0:
            self.fail("gamma must be positive", param, ctx)
        return g


_GAMMA = _GammaParamType()


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
    "--gamma",
    "gamma",
    type=_GAMMA,
    default=True,
    show_default=False,
    metavar="CURVE",
    help=(
        "Gamma/tone curve. "
        "'srgb' (default) = sRGB piecewise transfer function; "
        "'linear' = no gamma (will look dark); "
        "a positive number (e.g. 2.2) = simple power law v^(1/N). "
        "Ignored with --raw."
    ),
)
@click.option(
    "--exposure", "-e",
    type=float,
    default=0.0,
    show_default=True,
    metavar="STOPS",
    help=(
        "Exposure compensation in EV stops, applied before gamma. "
        "+1.0 doubles brightness; -1.0 halves it. Ignored with --raw."
    ),
)
@click.option(
    "--kernel", "-k",
    "kernel",
    type=click.Choice(["bilinear", "malvar", "menon", "ddfapd"], case_sensitive=False),
    default="bilinear",
    show_default=True,
    help=(
        "Demosaicing algorithm. 'bilinear' (default) needs no extra deps. "
        "'malvar' / 'menon' / 'ddfapd' require pip install shadow[demosaic]. "
        "Ignored with --raw or --half-res."
    ),
)
@click.option(
    "--no-ccm",
    is_flag=True,
    default=False,
    help=(
        "Skip factory color correction matrix (sensor RGB → XYZ → sRGB). "
        "When omitted the D65 forward_matrix from ColorProfile is applied when available. "
        "Ignored with --raw."
    ),
)
@click.option(
    "--no-awb",
    is_flag=True,
    default=False,
    help="Skip white-balance gains entirely.",
)
@click.option(
    "--no-orient",
    is_flag=True,
    default=False,
    help=(
        "Skip sensor orientation correction (flip_h / flip_v from the camera module proto). "
        "By default the image is flipped to match the sensor's physical mounting. "
        "Ignored with --raw."
    ),
)
@click.option(
    "--denoise", "denoise_kernel",
    type=click.Choice(["bm3d", "bilateral", "dncnn", "drunet"], case_sensitive=False),
    default=None,
    metavar="ALG",
    help=(
        "Denoising algorithm applied in linear light before colour correction. "
        "'bm3d' = Block Matching 3D, CPU (pip install shadow[denoise]). "
        "'bilateral' = GPU bilateral filter via kornia (pip install shadow[denoise-gpu]). "
        "'dncnn' = blind deep CNN; 'drunet' = sigma-aware residual U-Net "
        "(both via deepinv, pip install shadow[denoise-gpu]; download weights on first use). "
        "Ignored with --raw."
    ),
)
@click.option(
    "--denoise-sigma",
    type=float,
    default=0.05,
    show_default=True,
    metavar="SIGMA",
    help=(
        "Noise sigma for the denoiser (sigma_psd for BM3D). "
        "Range: 0.02 (subtle) to 0.15 (heavy). Only used with --denoise."
    ),
)
@click.option(
    "--awb-r",
    type=float,
    default=None,
    metavar="GAIN",
    help="Override the red AWB gain (e.g. 1.92). Ignored with --no-awb or --raw.",
)
@click.option(
    "--awb-b",
    type=float,
    default=None,
    metavar="GAIN",
    help="Override the blue AWB gain (e.g. 1.76). Ignored with --no-awb or --raw.",
)
def export(
    file: str,
    out_dir: str,
    camera: tuple[str, ...],
    fmt: str,
    raw: bool,
    half_res: bool,
    no_subtract_black: bool,
    gamma: bool | float,
    exposure: float,
    kernel: str,
    no_ccm: bool,
    no_awb: bool,
    no_orient: bool,
    denoise_kernel: str | None,
    denoise_sigma: float,
    awb_r: float | None,
    awb_b: float | None,
) -> None:
    """Export camera module images from an LRI file.

    Saves one image per camera module to OUT_DIR (default: current directory).
    Default output is 8-bit RGB PNG (debayered, AWB-corrected, sRGB gamma).
    Use --raw for 16-bit grayscale Bayer, or --format tiff for TIFF output.

    Output filenames: A1.png, B4_raw.tiff, etc.

    Examples:

        shadow export photo.lri ./out
        shadow export photo.lri ./out --raw --format tiff
        shadow export photo.lri ./out --camera B4 --half-res
        shadow export photo.lri ./out --exposure +1.5
        shadow export photo.lri ./out --gamma 2.2 --awb-r 2.0 --awb-b 1.8
        shadow export photo.lri ./out --gamma linear
    """
    lri = shadow.open_lri(file)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    images = _filter_cameras(lri.images, camera)
    if not images:
        console.print("[yellow]No matching camera images found.[/yellow]")
        sys.exit(1)

    subtract_black = not no_subtract_black
    apply_ccm = not no_ccm
    apply_awb = not no_awb
    apply_orientation = not no_orient
    demosaic_kernel = DemosaicKernel(kernel)
    denoise = DenoiseKernel(denoise_kernel) if denoise_kernel else None

    # Build AWB gains override if either channel was specified explicitly.
    awb_override: AwbGains | None = None
    if apply_awb and (awb_r is not None or awb_b is not None):
        # Use file gains as fallback for channels not overridden.
        file_gains = lri.metadata.awb_gains
        awb_override = AwbGains(
            r=awb_r if awb_r is not None else (file_gains.r if file_gains else 1.0),
            gr=file_gains.gr if file_gains else 1.0,
            gb=file_gains.gb if file_gains else 1.0,
            b=awb_b if awb_b is not None else (file_gains.b if file_gains else 1.0),
        )

    ext = "." + fmt.lower()
    suffix = "_raw" if raw else ""

    _print_settings(gamma, exposure, apply_awb, awb_override, apply_ccm, demosaic_kernel, apply_orientation, denoise, denoise_sigma, raw)

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

            kw = dict(
                raw=raw, half_res=half_res, subtract_black=subtract_black,
                apply_awb=apply_awb, awb_gains_override=awb_override,
                apply_ccm=apply_ccm, kernel=demosaic_kernel,
                gamma=gamma, exposure=exposure,
                apply_orientation=apply_orientation,
                denoise=denoise, denoise_sigma=denoise_sigma,
            )
            if fmt.lower() == "tiff":
                img.to_tiff(dest, **kw)
            else:
                img.to_png(dest, **kw)

            progress.advance(task)

    ref_name = lri.metadata.reference_camera.name if lri.metadata.reference_camera else "—"
    console.print(
        f"[green]Exported {len(images)} image(s) to[/green] {out}"
        f"  [dim](reference: {ref_name})[/dim]"
    )


def _print_settings(
    gamma: bool | float,
    exposure: float,
    apply_awb: bool,
    awb_override: AwbGains | None,
    apply_ccm: bool,
    kernel: DemosaicKernel,
    apply_orientation: bool,
    denoise: DenoiseKernel | None,
    denoise_sigma: float,
    raw: bool,
) -> None:
    if raw:
        return
    parts: list[str] = []
    if kernel != DemosaicKernel.BILINEAR:
        parts.append(f"kernel {kernel.value}")
    if exposure != 0.0:
        parts.append(f"exposure {exposure:+.2f} EV")
    if not apply_awb:
        parts.append("AWB off")
    elif awb_override is not None:
        parts.append(f"AWB R={awb_override.r:.3f} B={awb_override.b:.3f}")
    if not apply_ccm:
        parts.append("CCM off")
    if denoise is not None:
        parts.append(f"denoise {denoise.value} sigma={denoise_sigma:.3f}")
    if not apply_orientation:
        parts.append("orient off")
    if gamma is False:
        parts.append("gamma off")
    elif gamma is True:
        parts.append("gamma sRGB")
    else:
        parts.append(f"gamma {float(gamma):.2f}")
    if parts:
        console.print(f"  [dim]Settings: {', '.join(parts)}[/dim]")


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
