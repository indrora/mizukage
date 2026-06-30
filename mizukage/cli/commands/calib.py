"""shadow calib — dump Light L16 calibration directory data."""
from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()

# Known calibration file names in a lightcal directory.
_CAL_GEOMETRY   = "calibration.lri"
_CAL_ASIC       = "asic_calib_v1.lri"
_CAL_ZOOM       = "zoom_calib_v0.lri"
_CAL_HOTPIXEL   = "hotpixel.rec"


@click.command("calib")
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
@click.option("--json", "as_json", is_flag=True,
              help="Output full calibration data as JSON.")
def calib(directory: str, as_json: bool) -> None:
    """Dump calibration data from a Light L16 lightcal directory.

    Reads calibration.lri (geometry, colour, vignetting, sensor),
    asic_calib_v1.lri (ASIC-level geometry), zoom_calib_v0.lri (per-focus
    intrinsics), and hotpixel.rec (hot-pixel maps) from DIRECTORY.

    Each LRI/REC file is parsed as an LELR block stream; the embedded hot-pixel
    bitmaps are decoded and summarised (count + fraction) without including the
    full 13 MB/camera binary blobs in the output.
    """
    d = Path(directory)
    if as_json:
        _calib_json(d)
    else:
        _calib_print(d)


# ── Parsing helpers ────────────────────────────────────────────────────────────

def _parse_lri_blocks(path: Path) -> list[Any]:
    """Return a list of parsed LightHeader proto messages from an LELR file."""
    from shadow._block import iter_blocks
    import shadow._proto as _proto

    data = path.read_bytes()
    headers = []
    for block_start, hdr in iter_blocks(data):
        from shadow._block import BlockType
        if hdr.msg_type != BlockType.LIGHT_HEADER:
            continue
        proto_bytes = data[
            block_start + hdr.msg_offset :
            block_start + hdr.msg_offset + hdr.msg_len
        ]
        if not proto_bytes:
            continue
        try:
            headers.append(_proto.parse_light_header(proto_bytes))
        except Exception:
            pass
    return headers


def _parse_hotpixel(path: Path) -> dict[str, dict]:
    """Parse hotpixel.rec → per-camera hot-pixel statistics.

    Each camera's blob has a 20-byte header followed by zlib-compressed data:
      [0:4]  Unix timestamp (uint32 LE)
      [4:8]  padding / unknown (0)
      [8:12] compressed payload size (uint32 LE)
      [12:16] sensor width  (uint32 LE)
      [16:20] sensor height (uint32 LE)
      [20:]  zlib-deflated boolean bitmap (1 byte per pixel; 1 = hot)
    """
    from shadow._block import iter_blocks, BlockType
    import shadow._proto as _proto
    from shadow.proto import camera_id_pb2

    data = path.read_bytes()
    result: dict[str, dict] = {}

    for block_start, hdr in iter_blocks(data):
        if hdr.msg_type != BlockType.LIGHT_HEADER:
            continue
        proto_bytes = data[
            block_start + hdr.msg_offset :
            block_start + hdr.msg_offset + hdr.msg_len
        ]
        if not proto_bytes:
            continue
        try:
            msg = _proto.parse_light_header(proto_bytes)
        except Exception:
            continue

        for mc in msg.module_calibration:
            if not mc.HasField("hot_pixel_map"):
                continue
            cam_name = camera_id_pb2.CameraID.Name(mc.camera_id)
            hpm = mc.hot_pixel_map
            measurements = []
            for meas in hpm.data:
                abs_offset = block_start + int(meas.data_offset)
                raw = data[abs_offset : abs_offset + meas.data_size]
                try:
                    ts, _, comp_size, width, height = struct.unpack_from(
                        "<IIIII", raw, 0
                    )
                    bitmap = zlib.decompress(raw[20:])
                    n_hot = sum(bitmap)
                    total = len(bitmap)
                    measurements.append({
                        "timestamp": ts,
                        "width": width,
                        "height": height,
                        "hot_pixel_count": n_hot,
                        "hot_pixel_fraction": round(n_hot / total, 6) if total else 0.0,
                        "sensor_gain": meas.sensor_gain,
                        "sensor_exposure_us": meas.data_exposure,
                        "sensor_temperature_c": meas.sensor_temparature,
                        "pixel_variance": meas.pixel_variance if meas.HasField("pixel_variance") else None,
                        "threshold": meas.threshold if meas.HasField("threshold") else None,
                    })
                except Exception:
                    measurements.append({
                        "error": "failed to decode blob",
                        "data_size": meas.data_size,
                    })
            result[cam_name] = {"measurements": measurements}

    return result


def _lh_to_dict(lh) -> dict:
    """Convert a parsed LightHeader proto to a dict via MessageToDict."""
    from google.protobuf.json_format import MessageToDict
    return MessageToDict(lh, preserving_proto_field_name=True)


def _device_id(lh) -> tuple[str, str]:
    """Return (id_low_hex, id_high_hex) from a LightHeader."""
    lo = int(lh.image_unique_id_low) if lh.HasField("image_unique_id_low") else 0
    hi = int(lh.image_unique_id_high) if lh.HasField("image_unique_id_high") else 0
    # prefer device_unique_id_* fields
    if lh.HasField("device_unique_id_low"):
        lo = int(lh.device_unique_id_low)
    if lh.HasField("device_unique_id_high"):
        hi = int(lh.device_unique_id_high)
    return (f"{lo:016x}", f"{hi:016x}")


def _model(lh) -> str:
    return lh.device_model_name if lh.HasField("device_model_name") else ""


def _merge_geometry(headers: list) -> dict[str, list[dict]]:
    """Per-camera geometry from all focus bundles across LightHeader blocks."""
    from google.protobuf.json_format import MessageToDict
    out: dict[str, list[dict]] = {}
    for lh in headers:
        for mc in lh.module_calibration:
            if not mc.HasField("geometry"):
                continue
            from shadow.proto import camera_id_pb2
            cam = camera_id_pb2.CameraID.Name(mc.camera_id)
            geo = MessageToDict(mc.geometry, preserving_proto_field_name=True)
            out.setdefault(cam, []).append(geo)
    return out


def _merge_color(headers: list) -> dict[str, list[dict]]:
    """Per-camera colour calibration from all LightHeader blocks."""
    from google.protobuf.json_format import MessageToDict
    out: dict[str, list[dict]] = {}
    for lh in headers:
        for mc in lh.module_calibration:
            if not mc.color:
                continue
            from shadow.proto import camera_id_pb2
            cam = camera_id_pb2.CameraID.Name(mc.camera_id)
            for cc in mc.color:
                out.setdefault(cam, []).append(
                    MessageToDict(cc, preserving_proto_field_name=True)
                )
    return out


def _merge_vignetting(headers: list) -> dict[str, dict]:
    """Per-camera vignetting calibration (first occurrence wins)."""
    from google.protobuf.json_format import MessageToDict
    out: dict[str, dict] = {}
    for lh in headers:
        for mc in lh.module_calibration:
            if not mc.HasField("vignetting"):
                continue
            from shadow.proto import camera_id_pb2
            cam = camera_id_pb2.CameraID.Name(mc.camera_id)
            if cam not in out:
                out[cam] = MessageToDict(mc.vignetting, preserving_proto_field_name=True)
    return out


def _sensor_data(headers: list) -> dict | None:
    """Return first SensorData entry found across all LightHeader blocks."""
    from google.protobuf.json_format import MessageToDict
    for lh in headers:
        for sd in lh.sensor_data:
            return MessageToDict(sd, preserving_proto_field_name=True)
    return None


def _device_calibration(headers: list) -> dict | None:
    from google.protobuf.json_format import MessageToDict
    for lh in headers:
        if lh.HasField("device_calibration"):
            return MessageToDict(lh.device_calibration, preserving_proto_field_name=True)
    return None


# ── JSON output ────────────────────────────────────────────────────────────────

def _calib_json(d: Path) -> None:
    cal_path  = d / _CAL_GEOMETRY
    asic_path = d / _CAL_ASIC
    zoom_path = d / _CAL_ZOOM
    hp_path   = d / _CAL_HOTPIXEL

    cal_headers  = _parse_lri_blocks(cal_path)  if cal_path.exists()  else []
    asic_headers = _parse_lri_blocks(asic_path) if asic_path.exists() else []
    zoom_headers = _parse_lri_blocks(zoom_path) if zoom_path.exists() else []
    hot_pixels   = _parse_hotpixel(hp_path)     if hp_path.exists()   else {}

    # Device identity from the first available block
    device_id_lo, device_id_hi = ("", "")
    device_model = ""
    for lh in (cal_headers + asic_headers + zoom_headers):
        lo, hi = _device_id(lh)
        if lo != "0" * 16:
            device_id_lo, device_id_hi = lo, hi
            device_model = _model(lh) or device_model
            break

    geometry   = _merge_geometry(cal_headers)
    color      = _merge_color(cal_headers)
    vignetting = _merge_vignetting(cal_headers)
    sensor     = _sensor_data(cal_headers)
    dev_cal    = _device_calibration(cal_headers)

    asic_geometry = _merge_geometry(asic_headers)
    zoom_geometry = _merge_geometry(zoom_headers)

    # Merge everything per camera
    all_cams = sorted(
        set(geometry) | set(color) | set(vignetting) | set(hot_pixels)
        | set(asic_geometry) | set(zoom_geometry)
    )
    cameras: dict[str, dict] = {}
    for cam in all_cams:
        entry: dict[str, Any] = {}
        if cam in geometry:
            entry["geometry"] = geometry[cam]
        if cam in color:
            entry["color"] = color[cam]
        if cam in vignetting:
            entry["vignetting"] = vignetting[cam]
        if cam in hot_pixels:
            entry["hot_pixels"] = hot_pixels[cam]
        if cam in asic_geometry:
            entry["asic_geometry"] = asic_geometry[cam]
        if cam in zoom_geometry:
            entry["zoom_geometry"] = zoom_geometry[cam]
        cameras[cam] = entry

    out: dict[str, Any] = {
        "directory": str(d),
        "device": {
            "id_low":  device_id_lo,
            "id_high": device_id_hi,
            "model":   device_model,
        },
    }
    if sensor:
        out["sensor"] = sensor
    if dev_cal:
        out["device_calibration"] = dev_cal
    out["cameras"] = cameras

    click.echo(json.dumps(out, indent=2))


# ── Human-readable output ──────────────────────────────────────────────────────

def _mat3_to_tuple(m: dict) -> tuple[float, ...]:
    """Flatten a Matrix3x3F dict (x00..x22) to a 9-tuple."""
    return tuple(m.get(f"x{r}{c}", 0.0) for r in range(3) for c in range(3))


def _calib_print(d: Path) -> None:
    cal_path  = d / _CAL_GEOMETRY
    asic_path = d / _CAL_ASIC
    zoom_path = d / _CAL_ZOOM
    hp_path   = d / _CAL_HOTPIXEL

    cal_headers  = _parse_lri_blocks(cal_path)  if cal_path.exists()  else []
    asic_headers = _parse_lri_blocks(asic_path) if asic_path.exists() else []
    zoom_headers = _parse_lri_blocks(zoom_path) if zoom_path.exists() else []
    hot_pixels   = _parse_hotpixel(hp_path)     if hp_path.exists()   else {}

    # ── Device / sensor header ──────────────────────────────────────────────
    device_id_lo, device_id_hi = ("—", "—")
    device_model = "—"
    for lh in (cal_headers + asic_headers + zoom_headers):
        lo, hi = _device_id(lh)
        if lo != "0" * 16:
            device_id_lo, device_id_hi = lo, hi
            device_model = _model(lh) or device_model
            break

    sensor = _sensor_data(cal_headers)
    sensor_type = sensor["type"] if sensor else "—"
    black = sensor["data"]["black_level"] if sensor and "data" in sensor else "—"
    white = sensor["data"]["white_level"] if sensor and "data" in sensor else "—"

    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="bold cyan", no_wrap=True)
    summary.add_column()
    summary.add_row("Directory", str(d))
    summary.add_row("Device model", device_model)
    summary.add_row("Device ID", f"{device_id_lo}:{device_id_hi}")
    summary.add_row("Sensor", sensor_type)
    summary.add_row("Black / white", f"{black} / {white}")
    files_found = ", ".join(
        p.name for p in [cal_path, asic_path, zoom_path, hp_path] if p.exists()
    )
    summary.add_row("Files", files_found or "—")
    console.print(Panel(summary, title="[bold]Calibration[/bold]", border_style="blue"))

    # ── Per-camera calibration table ───────────────────────────────────────
    geometry   = _merge_geometry(cal_headers)
    color      = _merge_color(cal_headers)

    all_cams = sorted(
        set(geometry) | set(color) | set(hot_pixels) | set(_merge_geometry(asic_headers))
    )

    if not all_cams:
        console.print("[yellow]No per-camera calibration data found.[/yellow]")
        return

    tbl = Table(title="Per-camera calibration", box=box.SIMPLE_HEAD, show_lines=False)
    tbl.add_column("Camera", style="bold")
    tbl.add_column("fx / fy", justify="right")
    tbl.add_column("cx / cy", justify="right")
    tbl.add_column("Mirror", justify="center")
    tbl.add_column("Color illum.", justify="left")
    tbl.add_column("Hot pixels", justify="right")
    tbl.add_column("HP frac.", justify="right")

    for cam in all_cams:
        # Intrinsics from first focus bundle of primary geometry
        fx = fy = cx = cy = float("nan")
        mirror_type = "—"
        if cam in geometry:
            geo0 = geometry[cam][0]
            mirror_type = geo0.get("mirror_type", "NONE")
            bundles = geo0.get("per_focus_calibration", [])
            for b in bundles:
                intr = b.get("intrinsics", {}).get("k_mat", {})
                if intr:
                    fx = intr.get("x00", float("nan"))
                    fy = intr.get("x11", float("nan"))
                    cx = intr.get("x02", float("nan"))
                    cy = intr.get("x12", float("nan"))
                    break

        # Colour illuminants
        illums = sorted({cc.get("type", "?") for cc in color.get(cam, [])})
        illum_str = ", ".join(illums) if illums else "—"

        # Hot pixels
        hp_count = hp_frac = "—"
        if cam in hot_pixels:
            meas = hot_pixels[cam].get("measurements", [])
            if meas and "hot_pixel_count" in meas[0]:
                hp_count = f"{meas[0]['hot_pixel_count']:,}"
                hp_frac  = f"{meas[0]['hot_pixel_fraction']:.2%}"

        mirror_label = {"NONE": "fixed", "GLUED": "glued", "MOVABLE": "movable"}.get(
            mirror_type, mirror_type
        )

        fx_str = f"{fx:.1f} / {fy:.1f}" if fx == fx else "—"
        cx_str = f"{cx:.1f} / {cy:.1f}" if cx == cx else "—"
        tbl.add_row(cam, fx_str, cx_str, mirror_label, illum_str, hp_count, hp_frac)

    console.print(tbl)

    # ── Notes on what's available ───────────────────────────────────────────
    notes: list[str] = []
    if cal_path.exists():
        vigs = set(_merge_vignetting(cal_headers))
        if vigs:
            notes.append(f"vignetting calibration for {len(vigs)} cameras")
        dc = _device_calibration(cal_headers)
        if dc:
            notes.append(f"device calibration ({', '.join(dc)})")
    if asic_path.exists():
        notes.append(f"ASIC geometry ({asic_path.name})")
    if zoom_path.exists():
        notes.append(f"zoom/focus geometry ({zoom_path.name})")
    if notes:
        console.print(f"  [dim]Also available: {'; '.join(notes)}[/dim]\n")
