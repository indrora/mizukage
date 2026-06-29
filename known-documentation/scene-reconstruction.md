# Scene Reconstruction from L16 Multi-Camera Data

This document surveys 2D and 3D scene reconstruction techniques applicable to
the Light L16's 16-camera array, describes what the factory calibration data
provides and where it falls short, and assesses the practical feasibility of
each approach.

---

## The L16 as a reconstruction platform

The L16 captures 16 simultaneous images from cameras at three focal lengths,
all physically fixed to the same chassis:

| Array | Cameras | Focal length | fx (px) | Mirror |
|-------|---------|--------------|---------|--------|
| A | A1–A5 | 28 mm | ~3376 | Fixed |
| B | B1–B5 | 70 mm | ~8284 | Movable (OIS) |
| C | C1–C6 | 150 mm | ~18700 | Mostly movable |

The simultaneous capture avoids motion blur across views and eliminates
inter-view temporal inconsistency — a significant advantage over video-based
reconstruction that collects views by moving a single camera over time.

The device body is approximately 140 × 70 × 19 mm.  Within each array the
camera-to-camera baselines are on the order of 10–30 mm.  Inter-array baselines
(A to C centre) are larger (~50 mm horizontally).

---

## What the calibration data provides

From `calibration.lri` (see `calibration-data-guide.md`):

| Data | Available | Notes |
|------|-----------|-------|
| Intrinsics **K** (fx, fy, cx, cy) | ✓ | Per-camera, per-focus bundle |
| Radial distortion (5-coeff) | ✓ | Per-camera, normalised to 4160 px |
| Distortion centre | ✓ | Independently fitted from principal point |
| CRA model | ✓ | 100-sample (r_mm, angle_deg) table |
| Vignetting grid | ✓ | 17×13, some cameras focus-dependent |
| **Extrinsics** (R, t between cameras) | ✗ | Field exists in proto; empty on this device |
| Temperature-dependent intrinsics | ✗ | Field exists; only one temperature sample |

The `CalibrationFocusBundle.extrinsics` message has a `CanonicalFormat` with
`rotation` (3×3) and `translation` (Point3F), as well as `stereo_error` and
`reprojection_error` fields — the factory calibration rig clearly measured
these, but the data is not present in this device's files.  It may be present
in firmware builds with stereo processing enabled, or in the Lumen software
runtime.

From `shadow/__init__.py` / `LrisFile`:

The LRIS sidecar (produced by Lumen) provides a **195 × 260** quantized
disparity map for the reference camera view.  This is the only source of
metric depth currently available through the library.

---

## 2D reconstruction

### Depth / disparity from LRIS

The simplest path: use the Lumen-generated disparity map directly.

```python
lri  = shadow.open_lri("photo.lri")
lris = shadow.open_lris("photo.lris")
# lris.depth_map: int32 (195, 260), negative = invalid
# Convert disparity d to metric depth:
#   Z = (fx * baseline) / d
# where baseline and d are in the same pixel/mm units.
```

**Resolution:** 195×260 is very coarse (≈1/16 of the full 3120×4160 sensor).
Suitable for Lumen's computational refocus but insufficient for precision depth
or 3D work.

**Accuracy:** unknown; Lumen's pipeline is closed-source.  The `disparity_range`
property returns the raw quantized range; there is no documented scale factor.

### Pairwise stereo within a focal-length array

Within any array (e.g., B1–B5 at 70 mm) all cameras share the same approximate
intrinsics.  If extrinsics are recovered (see §3.1 below), pairwise stereo
using standard algorithms (SGM, RAFT-Stereo, CREStereo) becomes possible.

The intra-array baselines (~10–30 mm at 70 mm focal length) give a
depth-resolution product roughly matching what a 1/2.5" phone sensor can achieve
at 1× zoom — adequate for objects 0.5–5 m away.

Intra-array stereo is the most robust starting point because all cameras share
the same focal length, making rectification and disparity-to-depth conversion
straightforward.

### Cross-focal-length stereo (A ↔ B ↔ C)

Stereo between arrays at different focal lengths is possible but non-trivial:
- The scale factor between A and C is ~5.5×, so matched features appear at
  very different pixel sizes.
- Rectification must account for the different K matrices.
- The CRA model is important here: the chief ray angles differ across arrays and
  affect which regions of the scene are common to two cameras.

Cross-array stereo is most useful for **long-range depth** (beyond 5 m), where
the wider B/C baseline is needed for useful parallax.  For close objects the
A↔A or B↔B intra-array pairs are better conditioned.

### Synthetic aperture / light-field imaging

The 16 cameras can be interpreted as a sparse, irregular aperture sample.  For
a static scene, combining the 16 views with known extrinsics allows:

- **Synthetic refocus**: shift-and-add the registered views.  Lumen's
  computational refocus does exactly this.
- **Synthetic aperture photography**: views through foreground occlusions that
  no single camera could see.
- **Bokeh estimation**: the multiple focal lengths give independent blur-circle
  estimates, which can be composed into a realistic shallow-depth-of-field
  rendering at any chosen focal distance.

The CRA model from calibration is needed for accurate ray directions; ignoring
it introduces vignetting-like artefacts near the image edges.

### Super-resolution via multi-view fusion

Even cameras within the same array are not co-located: their sensors sample the
scene from slightly different angles.  At close range, sub-pixel relative shifts
between views can be exploited for super-resolution.  The parallax is strongest
for the C-array (150 mm, smallest field of view) and near subjects.

---

## 3D reconstruction

### 3.1 The extrinsics gap — options for recovery

All 3D reconstruction techniques require per-camera rotation and translation
(extrinsics).  Four paths exist:

#### A. Structure from Motion (COLMAP) on the 16 views

COLMAP's SfM pipeline can recover relative camera poses from feature matches
across the 16 simultaneously-captured images with **no motion required**.

```
colmap feature_extractor --image_path images/ --camera_model OPENCV \
    --camera_params "8284,8284,2073,1551"  # K for B cameras
colmap exhaustive_matcher --image_path images/
colmap mapper --image_path images/ --output_path sparse/
```

**Advantages:**
- Known intrinsics (from calibration) can be provided as priors, dramatically
  improving SfM convergence.
- The wide range of focal lengths (28–150 mm) provides rich overlapping coverage.
- Distortion coefficients map directly to COLMAP's `OPENCV` camera model
  (k1, k2, 0, 0, k3 for the 5-coeff radial polynomial, skipping p1/p2).

**Disadvantages:**
- 16 views of the same scene taken simultaneously may not have enough feature
  overlap across very different focal lengths.
- The A-array (28 mm, wide) and C-array (150 mm, tele) cover very different
  angular extents; feature matching A↔C is challenging.
- Works best when scene has rich texture; flat or low-texture scenes fail.

#### B. DUSt3R / MASt3R relative-pose estimation

DUSt3R (CVPR 2024) and its successor MASt3R recover relative camera poses
directly from image pairs using a transformer, without requiring explicit
feature matching.  MASt3R also produces dense 3D point maps.

For same-array pairs (B1↔B2, B3↔B4, etc.) this is likely the most robust
path, since the baseline is small and the focal lengths are identical.
Cross-array pairs are harder due to scale differences.

#### C. Mirror actuator model (B and C arrays only)

The `MovableMirrorFormat.mirror_system` proto describes the optical model of the
periscope mirror system used by B and C cameras.  If fully decoded, this
encodes how the mirror actuator position (`mirror_position` in
`CaptureMetadata.modules`) affects the effective camera ray bundle — which
implicitly contains the geometric relationship between the fixed chassis and the
folded optical path.

This path requires understanding the `MirrorSystem` proto fields, which are not
fully documented, but could yield sub-millimetre-accurate extrinsics from the
factory calibration if pursued.

#### D. Physical measurement from device specifications

The L16 teardown shows a regular grid of camera positions on the sensor board.
Approximate extrinsics (accurate to ~1 mm) can be derived from published
teardown photos and Light's patent filings, then refined via PnP against matched
features.

---

### 3.2 Multi-View Stereo (MVS)

Once extrinsics are available, full MVS with COLMAP (`colmap patch_match_stereo`)
or OpenMVS produces a dense point cloud at the reference camera resolution.

The L16 is well-suited for MVS because:
- All 16 images are noise-calibrated (consistent gain model, identical sensor).
- Vignetting correction from `calibration.lri` makes photometric consistency
  across views much better than uncalibrated multi-camera rigs.
- The three focal lengths give MVS algorithms a wide range of angular
  baselines to work with.

Apply image corrections before feeding to MVS:
1. Hot-pixel correction (interpolate before demosaic)
2. Vignetting correction (17×13 grid, bilinear)
3. Lens undistortion (5-coeff radial, OpenCV `undistortImage`)

The CRA model can additionally be used to correct photometric angular fall-off
per-ray, which matters for MVS photometric consistency at image borders.

---

### 3.3 NeRF (Neural Radiance Fields)

NeRF trains a continuous volumetric scene representation from a set of posed
images.  The L16's 16 simultaneous views make it a good fit for **single-capture
NeRF**: a full scene model from one press of the shutter, no object or camera
motion needed.

**Inputs (all available or derivable):**
- Camera intrinsics K: from calibration
- Camera extrinsics: from SfM/DUSt3R (see §3.1)
- Image: demosaiced, vignetting-corrected, undistorted exports from `shadow`

**Variant recommendations:**
- **Instant-NGP** (NVIDIA): fast hash-grid NeRF, seconds-to-minutes per scene.
  Handles the wide FOV mix (A+B+C) well since it's view-independent.
- **Nerfacto** (nerfstudio): production-quality, handles per-camera appearance
  conditioning — useful here because the 16 cameras have subtly different colour
  responses even after CCM.  Feed each camera's exposure and gain metadata as
  appearance embeddings.
- **Zip-NeRF**: better handling of varying focal lengths than vanilla NeRF.

**Per-camera appearance conditioning:**  
Even after forward-matrix colour correction, the 16 cameras differ in white
balance, vignetting residuals, and relative brightness.  NeRF variants that
accept a per-image appearance embedding (Nerfacto, NeRF-W) should model each
of the 16 cameras separately rather than treating them as interchangeable.

---

### 3.4 3D Gaussian Splatting (3DGS)

3DGS (Kerbl et al., SIGGRAPH 2023) represents scenes as a cloud of oriented
Gaussians and renders with differentiable rasterisation.  It is faster to train
and render than NeRF, with comparable quality.

**Input format (same as NeRF + COLMAP-sparse output):**

```
<scene>/
  input/            ← 16 undistorted images, one per camera
  sparse/0/
    cameras.bin     ← per-camera K (from calibration)
    images.bin      ← per-camera R, t (from SfM or DUSt3R)
    points3D.bin    ← sparse point cloud (from SfM, optional but helpful)
```

The official 3DGS implementation reads COLMAP output directly.  The key mapping:

| 3DGS field | Source |
|-----------|--------|
| `FovX`, `FovY` | `2 * atan(width / (2 * fx))` from K |
| Undistorted images | `shadow export` + OpenCV `undistortImage` |
| `R`, `T` | COLMAP mapper output |
| Initial point cloud | COLMAP sparse reconstruction |

**Multi-focal-length considerations:**

3DGS initialises Gaussians from the SfM sparse cloud and progressively
densifies them.  With cameras spanning 28–150 mm, the densification process
should be tuned:

- Set `--images_extension` and `--resolution` per-camera to avoid downscaling
  the tele C-array (its 4160×3120 at 150 mm carries the finest scene detail).
- The 28 mm A-array covers the widest sky/background area; consider splitting
  them from the B/C arrays in the COLMAP reconstruction and providing them as
  auxiliary views.

**Calibration-informed initialisation:**

The vignetting-corrected, undistorted images fed to 3DGS should be prepared as:

```python
# For each camera:
img_raw   = lri.image_for_camera(cam_id)
img_float = img_raw.to_debayered_numpy(apply_awb=True)
img_float *= vig_factor(cam_id)          # vignetting correction
img_undist = cv2.undistort(img_float, K, dist_coeffs)  # radial correction
img_u8    = (img_float * 255).clip(0,255).astype(np.uint8)
```

The `relative_brightness` scalar from vignetting calibration normalises the
exposure across arrays so 3DGS's per-Gaussian colour is not biased by
camera-to-camera brightness differences.

**Known limitation — movable mirrors:**

B and C cameras with movable mirrors introduce a focus-dependent ray shift.
If the mirror position was not at the calibration reference position during the
capture, the effective principal point shifts slightly.  The
`AngleOpticalCenterMapping` in the geometric calibration proto
(`center_start`, `center_end`, `angle_offset`, `t_scale`, `t_offset`) encodes
this shift.  For most captures this is a sub-pixel effect and can be ignored;
for precision reconstruction it should be applied before undistortion.

---

## Summary and recommended pipeline

```
1. Correct each image
   ├─ hot-pixel correction (hotpixel.rec bitmap, same-channel interpolation)
   ├─ vignetting correction (calibration.lri 17×13 grid)
   └─ lens undistortion (calibration.lri 5-coeff radial)

2. Recover extrinsics
   ├─ Option A: COLMAP SfM with calibration K as priors (works if scene has texture)
   └─ Option B: DUSt3R/MASt3R on same-array pairs → global alignment

3. Choose reconstruction target
   ├─ Depth map only: pairwise SGM stereo on B1↔B5 (largest intra-array baseline)
   ├─ Dense point cloud: COLMAP patch_match_stereo on all 16 views
   ├─ Neural scene + novel view synthesis: Nerfacto (nerfstudio) with per-camera appearance
   └─ Fast scene + real-time rendering: 3DGS from COLMAP sparse output

4. LRIS as cross-check
   └─ Lumen's 195×260 disparity map provides a coarse depth prior to validate
      or seed step 2/3, even if its resolution is insufficient for final output.
```

The biggest practical bottleneck is step 2.  Once extrinsics are recovered for
a given device unit and stored, they can be reused for all captures from that
device (the camera positions are fixed by the chassis).  Recovering them once
from a dedicated calibration scene (textured flat wall, or an aruco board)
would unlock the full reconstruction pipeline for all subsequent shots.
