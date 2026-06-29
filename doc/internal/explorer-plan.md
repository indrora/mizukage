# Plan: shadow-explorer — Dear ImGui calibration and LRI viewer

A standalone interactive tool (`shadow explore`) that lets you browse LRI
metadata, inspect per-camera images, and visualise calibration data from a
lightcal directory — all in a single resizable window.

---

## Dependency

[DearPyGui](https://github.com/hoffstadt/DearPyGui) is the standard Python
binding for Dear ImGui. It ships its own Dear ImGui build (no system lib
needed), has a stable API since 1.x, and supports Windows/macOS/Linux.

```toml
# pyproject.toml
[project.optional-dependencies]
explorer = ["dearpygui>=1.11"]
```

Install: `pip install 'shadow[explorer]'`

---

## New files

| Path | Purpose |
|------|---------|
| `shadow/cli/commands/explore.py` | Click command entry point |
| `shadow/explorer/__init__.py` | Window/panel orchestration |
| `shadow/explorer/_lri_panel.py` | LRI metadata + camera table |
| `shadow/explorer/_image_panel.py` | Live image preview with controls |
| `shadow/explorer/_calib_panel.py` | Calibration data visualisation |

Register in `shadow/cli/main.py` alongside `info`, `export`, etc.:
```python
from shadow.cli.commands.explore import explore
cli.add_command(explore)
```

---

## CLI entry point

```
shadow explore [FILE] [--calib DIR]

  FILE    Optional LRI file to open on launch.
  --calib Lightcal directory (calibration.lri, hotpixel.rec, etc.)
          If omitted, the tool tries to auto-discover a sibling lightcal/
          directory next to FILE.
```

The explore command must be a lazy import so that users without `dearpygui`
installed get a clear error message, not an `ImportError` at CLI startup:

```python
@click.command("explore")
@click.argument("file", required=False, ...)
@click.option("--calib", "calib_dir", ...)
def explore(file, calib_dir):
    """Interactively explore LRI metadata and calibration data."""
    try:
        import dearpygui.dearpygui as dpg
    except ImportError:
        raise click.UsageError(
            "shadow[explorer] is not installed. "
            "Run: pip install 'shadow[explorer]'"
        )
    from shadow.explorer import run_explorer
    run_explorer(file, calib_dir)
```

---

## Window layout

```
┌─────────────────────────────────────────────────────────────────────────┐
│  shadow explorer                                    [_][□][X]           │
├───────────────┬─────────────────────────────┬───────────────────────────┤
│  Files        │  Image preview              │  Properties               │
│               │                             │                           │
│  [Open LRI…]  │  ┌─────────────────────┐   │  File: L16_00009.lri      │
│  [Open Cal…]  │  │                     │   │  Size: 170.6 MB           │
│               │  │   (camera image)    │   │  Images: 11               │
│  Cameras      │  │                     │   │  Focal: 149 mm            │
│  ──────────   │  │                     │   │  Ref cam: B4              │
│  ○ A1         │  └─────────────────────┘   │  Orient: landscape        │
│  ○ A2         │                             │  AWB: R=1.92 B=1.76       │
│  ● B4 (ref)   │  Camera:  [B4        ▼]    │  On tripod: no            │
│  ○ B5         │  Exposure [━━●━━━━━━━━] 0.0│                           │
│  ○ C1         │  Gamma    [sRGB / Lin ▼]   │  ── Per-camera ──         │
│  …            │  AWB      [✓]              │  B4: 4160×3120 BGGR       │
│               │  CCM      [✓]              │  Gain: 3.875  Exp: 15.5ms │
│               │  Orient   [✓]              │  Flip: —                  │
│               │  [Export PNG]              │                           │
│               │                             │                           │
├───────────────┴─────────────────────────────┴───────────────────────────┤
│  Calibration   [Vignetting] [Hot pixels] [Noise model] [Geometry] [Color]│
│                                                                          │
│  (tab content — see §Calibration panel below)                           │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Panels

### Files panel (left sidebar)

- `[Open LRI…]` button → file dialog (`dpg.add_file_dialog`)
- `[Open Cal…]` button → directory dialog
- Camera list: radio buttons, one per `lri.images`, reference camera marked
- Selecting a camera updates the image preview and per-camera properties

### Image preview panel (centre)

- Renders the current camera's debayered image as a DearPyGui texture.
  `dpg.add_raw_texture` accepts a flat float32 RGBA buffer.
- Controls below the image:
  - **Camera** dropdown (mirrors the sidebar selection)
  - **Exposure** slider (−3.0 to +3.0 EV, step 0.1)
  - **Gamma** combo: sRGB / Linear / 2.2
  - **AWB** / **CCM** / **Orient** checkboxes
  - **[Export PNG]** button — saves current view to disk
- Image re-renders asynchronously when controls change (use a dirty flag;
  render on the next frame, not in the callback, to avoid re-entrancy).
- Resolution: render at half-res by default (`half_res=True`) for speed;
  add a `[Full res]` toggle.

**Texture pipeline:**
```python
rgb8 = img.to_debayered_numpy(half_res=True, ...)  # (H, W, 3) uint16 after normalise
rgba_f32 = np.zeros((H, W, 4), dtype=np.float32)
rgba_f32[:, :, :3] = rgb8.astype(np.float32) / 255.0
rgba_f32[:, :, 3] = 1.0
dpg.set_value(texture_tag, rgba_f32.flatten().tolist())
```

### Properties panel (right)

- Top section: file-level metadata (same fields as `shadow info` human output)
- Per-camera section: updates when a camera is selected in the sidebar
  - Sensor model, format, size, CFA, gain, exposure, flip flags

### Calibration panel (bottom, tabbed)

Tabs are only populated when a calibration directory is loaded. Each tab
renders its content only when selected (lazy render).

#### Vignetting tab

- Camera selector combo
- 17×13 grid rendered as a heatmap using DearPyGui's `add_heat_series` on a
  `add_plot`. Colour map: viridis or inferno (white = high correction factor).
- If C-array camera: hall-code selector to switch between the 4 focus entries.
- Display scalar: "relative brightness: 1.62" under the heatmap.

#### Hot pixels tab

- Camera selector combo
- Hot-pixel bitmap rendered as a downsampled greyscale image (full 4160×3120
  is too large; downsample to ≤512px wide before uploading as texture).
- Statistics below: "X hot pixels (Y%)" from the 20-byte header.
- Colour: hot pixels in red, normal in dark grey.

#### Noise model tab

- Line plot of σ vs. gain code (green channel `sqrt(a * mid_signal + b)`)
  using `dpg.add_line_series` on a plot.
- All 28 gain entries (100–775) on the x-axis.
- Vertical marker at the current camera's `analog_gain * 100`.
- Secondary y-axis: raw `a` coefficient (shot noise slope).

#### Geometry tab

- Camera selector combo
- Table: focus bundle, hall code, fx, fy, cx, cy
- Distortion coefficients k1–k5 in a grid
- Distortion centre (px, py)
- Mirror type label

#### Color tab

- Camera selector combo
- Three illuminant tabs (D65 / A / F11)
- 3×3 forward matrix rendered as a colour-coded table (negative=red,
  positive=green, near-zero=grey)
- rg_ratio, bg_ratio neutral point
- Per-illuminant: "Closest to capture AWB? ✓/—" indicator

---

## Async image rendering

DearPyGui runs on the main thread. Image decoding (raw unpack + demosaic) is
CPU-heavy. To keep the UI responsive:

- Run the decode in a `threading.Thread`.
- When complete, set a flag; the next `dpg.render_dearpygui_frame()` call
  uploads the texture and clears the flag.
- Show a spinner (`dpg.add_loading_indicator`) while rendering.
- Cancel in-flight renders if the user switches cameras before it finishes.

```python
import threading
_render_lock = threading.Lock()
_pending: threading.Thread | None = None

def _request_render(img, params):
    global _pending
    with _render_lock:
        if _pending and _pending.is_alive():
            _pending = None  # old thread will check a cancel flag
    _pending = threading.Thread(target=_do_render, args=(img, params), daemon=True)
    _pending.start()
```

---

## IMU / orientation indicator

In the Properties panel, below the orientation label, render a small 2D
compass rose using `dpg.draw_circle` / `dpg.draw_arrow` on a `drawlist`:

- Draw device outline (rectangle, ~60×40 px) rotated by the detected angle.
- Arrow pointing "up" in the corrected orientation.
- Label: "landscape" / "portrait (top-left)" etc.

---

## Implementation order

1. Skeleton: window opens, file dialog works, metadata displays in Properties.
2. Image preview: texture upload, camera selection, exposure/gamma controls.
3. Calibration panel: vignetting and hot-pixel tabs (most visually useful).
4. Noise model plot.
5. Geometry and Color tabs.
6. IMU compass indicator.
7. Async rendering thread.

---

## Open questions

- **Platform:** DearPyGui requires a display server. On headless Linux, it
  needs an X11 stub (Xvfb) or a build with offscreen rendering. This is a
  development/desktop tool, not a server-side one — documented limitation.
- **Image size:** Full-res 4160×3120 float32 RGBA = ~49 MB per camera.
  Half-res is ~12 MB. The default should be half-res; full-res on demand.
- **Multiple LRI files:** v1 opens one file at a time. A future version could
  open a session with multiple LRI files for cross-capture comparison.
- **Export from explorer:** should `[Export PNG]` honour the calibration
  corrections (hot-pixel, vignetting) if a calib dir is loaded? Yes — this
  would be the cleanest UX and is a natural integration point for the
  calibration pipeline work landing in the parallel agent branches.
