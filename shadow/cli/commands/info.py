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
    if as_json:
        _info_lri_json(path)
        return

    lri = shadow.open_lri(str(path))
    meta = lri.metadata

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
    if meta.orientation is not None:
        _ORIENT_LABELS = {
            0: "landscape",
            1: "portrait (top-right)",
            2: "portrait (top-left)",
            7: "landscape (inverted)",
        }
        from shadow._types import Orientation
        summary.add_row(
            "Orientation",
            _ORIENT_LABELS.get(meta.orientation.value, meta.orientation.name),
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


def _info_lri_json(path: Path) -> None:
    """Emit a full-fidelity JSON document of all LELR proto blocks in an LRI file.

    Each block's proto message is serialized via google.protobuf.json_format
    MessageToDict, which faithfully converts nested messages, repeated fields,
    packed float arrays, and enum values (as their string names).  Only fields
    actually present in the file are included; proto2 optional defaults are omitted.

    Block-level housekeeping fields (_block_offset, _block_size, _proto_size) are
    injected alongside the proto payload so callers can locate data in the raw file.
    """
    from google.protobuf.json_format import MessageToDict
    import shadow._proto as _proto

    file_bytes = path.read_bytes()

    light_headers: list[dict] = []
    view_prefs: list[dict] = []
    gps_blocks: list[dict] = []
    unknown_blocks: list[dict] = []

    for block_start, hdr in iter_blocks(file_bytes):
        proto_bytes = file_bytes[
            block_start + hdr.msg_offset :
            block_start + hdr.msg_offset + hdr.msg_len
        ]
        if not proto_bytes:
            continue

        block_meta = {
            "_block_offset": block_start,
            "_block_size": hdr.block_length,
            "_proto_size": hdr.msg_len,
        }

        try:
            if hdr.msg_type == BlockType.LIGHT_HEADER:
                msg = _proto.parse_light_header(proto_bytes)
                d = MessageToDict(
                    msg,
                    preserving_proto_field_name=True,
                )
                d.update(block_meta)
                light_headers.append(d)

            elif hdr.msg_type == BlockType.VIEW_PREFERENCES:
                msg = _proto.parse_view_preferences(proto_bytes)
                d = MessageToDict(
                    msg,
                    preserving_proto_field_name=True,
                )
                d.update(block_meta)
                view_prefs.append(d)

            elif hdr.msg_type == BlockType.GPS_DATA:
                msg = _proto.parse_gps_data_proto(proto_bytes)
                d = MessageToDict(
                    msg,
                    preserving_proto_field_name=True,
                )
                d.update(block_meta)
                gps_blocks.append(d)

            else:
                unknown_blocks.append({
                    "type": hdr.msg_type.name,
                    **block_meta,
                })

        except Exception as exc:
            # Include corrupt/unrecognised blocks as error entries rather than
            # silently dropping them.
            unknown_blocks.append({
                "type": hdr.msg_type.name,
                "parse_error": str(exc),
                **block_meta,
            })

    out: dict = {
        "file": str(path),
        "size_bytes": path.stat().st_size,
        "light_headers": light_headers,
        "view_preferences": view_prefs,
    }
    if gps_blocks:
        out["gps_data"] = gps_blocks
    if unknown_blocks:
        out["unknown_blocks"] = unknown_blocks

    click.echo(json.dumps(out, indent=2))


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
