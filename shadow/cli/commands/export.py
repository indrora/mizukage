"""shadow export — save camera module images as PNG or TIFF."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Column

import shadow
from shadow._calib import compute_scalar_sigma, load_vst_model
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
    type=click.Choice(
        ["bilateral_spatial", "bm3d", "bilateral", "dncnn", "drunet"],
        case_sensitive=False,
    ),
    default=None,
    metavar="ALG",
    help=(
        "Denoising algorithm applied in linear light before colour correction. "
        "'bilateral_spatial' = pure-numpy spatial bilateral filter (no extra deps; "
        "slow on large images, ~10-30 s). "
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
    "--denoise-tile-size",
    type=int,
    default=512,
    show_default=True,
    metavar="PX",
    help=(
        "Tile size for DnCNN / DRUNet inference (pixels per side, default 512). "
        "Increase to 1024 or 2048 on high-VRAM GPUs (24 GB can handle the full image "
        "at 3120); reduce to 256 on 4-6 GB GPUs. Ignored for BM3D and bilateral."
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
@click.option(
    "--calib",
    "calib_dir",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    metavar="DIR",
    help=(
        "Path to a Light L16 lightcal directory. When provided: hot-pixel maps "
        "from hotpixel.rec are applied before demosaicing; when combined with "
        "--denoise, the factory VST noise model sets per-camera sigma automatically."
    ),
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
    denoise_tile_size: int,
    awb_r: float | None,
    awb_b: float | None,
    calib_dir: str | None,
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
        shadow export photo.lri ./out --camera B4 --calib images/lightcal
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

    # Load the factory VST noise model when a calibration directory is provided
    # alongside a denoising kernel.  An empty list means "model unavailable".
    vst_model = (
        load_vst_model(Path(calib_dir))
        if calib_dir is not None and denoise is not None
        else []
    )
    if calib_dir is not None and denoise is not None and not vst_model:
        console.print(
            "[yellow]Warning: no VST model found in calibration directory; "
            "falling back to --denoise-sigma.[/yellow]"
        )

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

    _print_settings(gamma, exposure, apply_awb, awb_override, apply_ccm, demosaic_kernel, apply_orientation, denoise, denoise_sigma, denoise_tile_size, raw, calib_dir)

    # Cache for hot-pixel maps: loaded once per camera_id to avoid re-parsing
    # the calibration file on every image in a multi-camera export.
    hp_cache: dict[CameraId, object] = {}

    desc_col = TextColumn(
        "{task.description}",
        table_column=Column(min_width=28, no_wrap=True),
    )

    with Progress(
        SpinnerColumn(),
        desc_col,
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        # Row 1: per-image stage — total set per image from pre-computed step count
        image_task = progress.add_task("", total=1)
        # Row 2: overall batch — one unit per image
        batch_task = progress.add_task("  [dim]batch[/dim]", total=len(images))

        for img in images:
            name = img.camera_id.name
            dest = out / f"{name}{suffix}{ext}"

            # When the factory VST noise model is available, use it to derive
            # a per-camera sigma from the actual capture analog gain.  This is
            # more accurate than the user-supplied scalar because it accounts
            # for the sensor's measured shot-noise and read-noise coefficients.
            effective_sigma = denoise_sigma
            if vst_model and denoise is not None:
                effective_sigma = compute_scalar_sigma(vst_model, img.analog_gain)
                console.print(
                    f"  [dim]Using VST sigma={effective_sigma:.3f} "
                    f"for camera {name} "
                    f"(analog_gain={img.analog_gain:.2f})[/dim]"
                )

            # Pre-compute the exact number of progress advances for this image
            # so the per-image bar shows a real fraction rather than a pulse.
            total_steps = _image_steps(img, half_res, raw, denoise, denoise_tile_size)
            progress.reset(image_task, total=total_steps)

            def on_step(stage: str, _n: str = name) -> None:
                progress.update(image_task, description=f"  [bold]{_n}[/bold] {stage}")

            def on_advance(n: int, _task=image_task) -> None:
                progress.advance(_task, n)

            # Resolve hot-pixel map for this camera (lazy, cached per camera_id).
            hot_pixel_map = None
            if calib_dir is not None:
                cam_id = img.camera_id
                if cam_id not in hp_cache:
                    from shadow._calib import load_hot_pixel_map
                    hp_cache[cam_id] = load_hot_pixel_map(Path(calib_dir), cam_id)
                hot_pixel_map = hp_cache[cam_id]

            kw = dict(
                raw=raw, half_res=half_res, subtract_black=subtract_black,
                apply_awb=apply_awb, awb_gains_override=awb_override,
                apply_ccm=apply_ccm, kernel=demosaic_kernel,
                gamma=gamma, exposure=exposure,
                apply_orientation=apply_orientation,
                denoise=denoise, denoise_sigma=effective_sigma,
                denoise_tile_size=denoise_tile_size,
                on_step=on_step, on_advance=on_advance,
                hot_pixel_map=hot_pixel_map,
            )
            if fmt.lower() == "tiff":
                img.to_tiff(dest, **kw)
            else:
                img.to_png(dest, **kw)

            progress.update(image_task, description=f"  [dim]{name} done[/dim]")
            progress.advance(batch_task)

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
    denoise_tile_size: int,
    raw: bool,
    calib_dir: str | None = None,
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
        tile_info = f" tile={denoise_tile_size}" if denoise in (DenoiseKernel.DNCNN, DenoiseKernel.DRUNET) else ""
        parts.append(f"denoise {denoise.value} sigma={denoise_sigma:.3f}{tile_info}")
    if not apply_orientation:
        parts.append("orient off")
    if gamma is False:
        parts.append("gamma off")
    elif gamma is True:
        parts.append("gamma sRGB")
    else:
        parts.append(f"gamma {float(gamma):.2f}")
    if calib_dir is not None:
        parts.append("hot-pixel correction on")
    if parts:
        console.print(f"  [dim]Settings: {', '.join(parts)}[/dim]")


def _image_steps(img, half_res: bool, raw: bool,
                 denoise: DenoiseKernel | None, denoise_tile_size: int) -> int:
    """Pre-compute the total progress advance count for one image.

    Each pipeline stage contributes its own op count:
      debayer          → 1
      bm3d / bilateral → 1
      dncnn / drunet   → number of tiles (depends on image size and tile_size)
      color correction → 1
      save             → 1
    raw export skips everything except save, so it returns 1.
    """
    from shadow._denoise import count_tiles

    if raw:
        return 1

    H = img.height // 2 if half_res else img.height
    W = img.width // 2 if half_res else img.width

    ops = 3  # debayer + color correction + save
    if denoise in (DenoiseKernel.DNCNN, DenoiseKernel.DRUNET):
        ops += count_tiles(H, W, denoise_tile_size)
    elif denoise is not None:
        ops += 1  # bm3d or bilateral: single op

    return ops


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
