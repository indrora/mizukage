"""RawImage — a single camera module's sensor data from an LRI file."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

from shadow._debayer import DemosaicKernel
from shadow._denoise import DenoiseKernel
from shadow._types import (
    AwbGains,
    BayerPattern,
    CameraId,
    ColorProfile,
    Illuminant,
    Orientation,
    RawFormat,
    SensorModel,
)

# 10-bit data after black-level subtract: usable range [0 .. 1023-black_level]
_10BIT_MAX = 1023

# XYZ D50 (DNG Profile Connection Space) → linear sRGB.
# The L16's forward_matrix maps white-balanced sensor RGB → XYZ D50, not D65.
# This matrix applies the Bradford D50→D65 chromatic adaptation implicitly.
_XYZ_D50_TO_SRGB_LINEAR = np.array([
    [ 3.1338561, -1.6168667, -0.4906146],
    [-0.9787684,  1.9161415,  0.0334540],
    [ 0.0719453, -0.2289914,  1.4052427],
], dtype=np.float32)


def _apply_forward_matrix(rgb: np.ndarray, forward_matrix: tuple[float, ...]) -> np.ndarray:
    """Apply factory forward_matrix (sensor RGB → XYZ D65) then XYZ → linear sRGB.

    rgb: float32 (H, W, 3) normalized linear sensor RGB
    forward_matrix: 9 floats, row-major, from ColorProfile

    The two-step matrix multiplication converts the sensor's native primaries
    to display-standard sRGB primaries. Output may fall outside [0..1] for
    highly saturated colours; the subsequent gamma function clips before encoding.
    """
    fm = np.array(forward_matrix, dtype=np.float32).reshape(3, 3)
    h, w = rgb.shape[:2]
    flat = rgb.reshape(-1, 3)
    xyz = flat @ fm.T                          # sensor RGB → XYZ D50
    srgb = xyz @ _XYZ_D50_TO_SRGB_LINEAR.T    # XYZ D50 → linear sRGB
    return srgb.reshape(h, w, 3)


def _apply_gamma(normalized: np.ndarray, gamma: bool | float) -> np.ndarray:
    """Apply a gamma/tone curve to a float32 array already normalised to [0..1].

    gamma=True  → full sRGB piecewise transfer function
    gamma=False → identity (linear, no encoding)
    gamma=float → simple power-law encoding: out = v^(1/gamma)
                  e.g. gamma=2.2 approximates sRGB without the linear toe

    Note: check True/False with `is` before any numeric comparison because
    bool is a subclass of int and True == 1 == 1.0 in Python.
    """
    v = np.clip(normalized, 0.0, 1.0).astype(np.float32)
    if gamma is True:
        return np.where(v <= 0.0031308, 12.92 * v, 1.055 * np.power(v, 1.0 / 2.4) - 0.055)
    if gamma is False or float(gamma) == 1.0:
        return v
    return np.power(v, 1.0 / float(gamma)).astype(np.float32)


@dataclass
class RawImage:
    """Raw sensor data for a single L16 camera module.

    Pixel data is read lazily from the source file bytes on demand.
    """

    # ── Public, user-visible fields ───────────────────────────────────────────
    camera_id: CameraId
    sensor_model: SensorModel
    width: int
    height: int
    raw_format: RawFormat
    bayer_r_row: int | None  # None = mono sensor
    bayer_r_col: int | None
    analog_gain: float
    exposure_ns: int
    digital_gain: float | None
    flip_h: bool
    flip_v: bool
    color_profiles: list[ColorProfile] = field(default_factory=list)
    awb_gains: AwbGains | None = None  # capture-level white-balance gains

    # ── Private: populated by LriFile, not intended for direct user access ────
    # kw_only=True so they don't break the positional-default ordering rule.
    _file_bytes: bytes = field(repr=False, compare=False, kw_only=True)
    _data_offset: int = field(repr=False, compare=False, kw_only=True)
    _row_stride: int = field(repr=False, compare=False, kw_only=True, default=0)
    _black_level: float = field(repr=False, compare=False, kw_only=True, default=64.0)
    # IMU-derived or ViewPreferences-derived device-hold orientation.
    # Back-filled by LriFile after all LELR blocks are parsed.
    _imu_orientation: Orientation | None = field(repr=False, compare=False, kw_only=True, default=None)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def cfa_pattern(self) -> BayerPattern | None:
        """CFA layout string (e.g. BGGR). None for mono sensors."""
        if self.bayer_r_row is None or self.bayer_r_col is None:
            return None
        index = self.bayer_r_col | (self.bayer_r_row << 1)
        return BayerPattern(index)

    @property
    def is_mono(self) -> bool:
        return self.sensor_model.is_mono

    @property
    def exposure_ms(self) -> float:
        return self.exposure_ns / 1_000_000.0

    @property
    def white_level(self) -> int:
        """Usable white level after black subtraction (10-bit data)."""
        return _10BIT_MAX - int(self._black_level)

    def color_profile(self, illuminant: Illuminant) -> ColorProfile | None:
        return next((p for p in self.color_profiles if p.illuminant == illuminant), None)

    # ── Numpy access ─────────────────────────────────────────────────────────

    def to_raw_numpy(self, *, subtract_black: bool = True) -> np.ndarray:
        """Unpack raw Bayer/mono pixel data → uint16 array (height, width).

        No demosaicing is performed. Values are in [0..1023] (10-bit).
        With subtract_black=True (default), the black level is subtracted and
        the result is clipped to [0..white_level].
        """
        from shadow._unpack import unpack_10bpp, decode_bjpg

        match self.raw_format:
            case RawFormat.PACKED_10BPP | RawFormat.PACKED_12BPP | RawFormat.PACKED_14BPP:
                arr = unpack_10bpp(
                    self._file_bytes,
                    self._data_offset,
                    self.width,
                    self.height,
                    self._row_stride,
                )
            case RawFormat.BAYER_JPEG:
                arr = decode_bjpg(
                    self._file_bytes,
                    self._data_offset,
                    self.width,
                    self.height,
                    self.bayer_r_row if self.bayer_r_row is not None else 1,
                    self.bayer_r_col if self.bayer_r_col is not None else 1,
                )
            case _:
                raise NotImplementedError(f"Unsupported raw format: {self.raw_format!r}")

        if subtract_black:
            bl = int(self._black_level)
            arr = np.clip(arr.astype(np.int32) - bl, 0, _10BIT_MAX - bl).astype(np.uint16)

        return arr

    def to_debayered_numpy(
        self,
        *,
        half_res: bool = False,
        subtract_black: bool = True,
        apply_awb: bool = True,
        awb_gains_override: AwbGains | None = None,
        kernel: DemosaicKernel = DemosaicKernel.BILINEAR,
    ) -> np.ndarray:
        """Demosaic the raw Bayer data to an RGB array.

        Returns float32 (height, width, 3) or (height/2, width/2, 3) if half_res.
        Values are in the same linear-light range as the input (e.g. [0..~959]
        after black subtraction). No gamma is applied here.

        apply_awb=True  → apply per-channel white-balance gains before demosaicing
        awb_gains_override → if provided, use these gains instead of self.awb_gains
                             (only meaningful when apply_awb=True)
        kernel          → demosaicing algorithm; see DemosaicKernel for choices.
                          MALVAR/MENON/DDFAPD require ``pip install shadow[demosaic]``.
                          half_res=True overrides to HALF regardless of kernel.
        """
        from shadow._debayer import debayer_half, debayer_bilinear, debayer_colour

        raw = self.to_raw_numpy(subtract_black=subtract_black)

        if self.is_mono or self.bayer_r_row is None:
            rgb = np.stack([raw, raw, raw], axis=2)
            return rgb.astype(np.float32)

        r_row = self.bayer_r_row
        r_col = self.bayer_r_col

        gains = None
        if apply_awb:
            gains = awb_gains_override if awb_gains_override is not None else self.awb_gains

        if gains is not None:
            b_row, b_col = 1 - r_row, 1 - r_col
            bayer = raw.astype(np.float32)
            bayer[r_row::2, r_col::2] *= gains.r
            bayer[r_row::2, b_col::2] *= gains.gr
            bayer[b_row::2, r_col::2] *= gains.gb
            bayer[b_row::2, b_col::2] *= gains.b
        else:
            bayer = raw.astype(np.float32)

        if half_res:
            return debayer_half(bayer, r_row, r_col)

        if kernel == DemosaicKernel.BILINEAR:
            return debayer_bilinear(bayer, r_row, r_col)

        return debayer_colour(bayer, r_row, r_col, kernel)

    # ── File export ───────────────────────────────────────────────────────────

    def _orient(self, arr: np.ndarray) -> np.ndarray:
        """Compose the 180° sensor-mount correction with the IMU device-hold rotation.

        All L16 sensors are physically mounted 180° upside-down.  On top of
        that, the user may hold the device in portrait or inverted landscape.
        Composing both into one numpy operation avoids an extra array copy.

        Derivation (array rotations, CCW positive):
          sensor 180°  +  NORMAL (landscape)       = 180°   → arr[::-1, ::-1]
          sensor 180°  +  ROT90_CCW (top-left)     = 90° CW → np.rot90(arr, k=-1)
          sensor 180°  +  ROT90_CW  (top-right)    = 90°CCW → np.rot90(arr, k=1)
          sensor 180°  +  ROT180 (upside-down)      = 0°     → arr (identity)
        """
        hold = self._imu_orientation
        if hold == Orientation.ROT90_CCW:
            return np.ascontiguousarray(np.rot90(arr, k=-1))
        if hold == Orientation.ROT90_CW:
            return np.ascontiguousarray(np.rot90(arr, k=1))
        if hold == Orientation.ROT180:
            return np.ascontiguousarray(arr)
        # NORMAL, None, or unhandled: 180° sensor-mount fix only
        return np.ascontiguousarray(arr[::-1, ::-1])

    def _export_rgb8(
        self,
        *,
        half_res: bool,
        subtract_black: bool,
        apply_awb: bool,
        awb_gains_override: AwbGains | None,
        apply_ccm: bool,
        kernel: DemosaicKernel,
        gamma: bool | float,
        exposure: float,
        apply_orientation: bool,
        denoise: DenoiseKernel | None = None,
        denoise_sigma: float = 0.05,
        denoise_tile_size: int = 512,
        on_step: Callable[[str], None] | None = None,
        on_advance: Callable[[int], None] | None = None,
    ) -> np.ndarray:
        """Shared debayer → normalise → denoise → CCM → exposure → gamma → orient → uint8."""
        _step = on_step if on_step is not None else lambda _: None
        _adv = on_advance if on_advance is not None else lambda n: None

        _step("debayering")
        white = self.white_level if subtract_black else _10BIT_MAX
        rgb = self.to_debayered_numpy(
            half_res=half_res,
            subtract_black=subtract_black,
            apply_awb=apply_awb,
            awb_gains_override=awb_gains_override,
            kernel=kernel,
        )
        normalized = (rgb / white).astype(np.float32)
        _adv(1)
        if denoise is not None:
            _step(f"denoising ({denoise.value})")
            from shadow._denoise import denoise_image
            # Clip before denoising: values must be in [0, 1] for BM3D.
            np.clip(normalized, 0.0, 1.0, out=normalized)
            normalized = denoise_image(
                normalized, denoise,
                sigma=denoise_sigma, tile_size=denoise_tile_size,
                on_advance=on_advance,
            )
        _step("color correction")
        if apply_ccm:
            # Clip to [0, 1] before the CCM. AWB gains > 1.0 can push saturated
            # channels above white_level; the forward_matrix's negative cross-terms
            # then collapse G (e.g. FM([1.92, 1.0, 1.76]) → sRGB G≈0.26 = pink).
            # Clipping first ensures saturated highlights render as white, not magenta.
            np.clip(normalized, 0.0, 1.0, out=normalized)
            # Prefer the D65 profile; fall back to any available illuminant.
            prof = self.color_profile(Illuminant.D65)
            if prof is None and self.color_profiles:
                prof = self.color_profiles[0]
            if prof is not None:
                normalized = _apply_forward_matrix(normalized, prof.forward_matrix)
        if exposure != 0.0:
            normalized *= 2.0 ** exposure
        normalized = _apply_gamma(normalized, gamma)
        result = (normalized * 255.0).clip(0, 255).astype(np.uint8)
        if apply_orientation:
            result = self._orient(result)
        _adv(1)
        return result

    def to_png(
        self,
        path: str | Path,
        *,
        raw: bool = False,
        half_res: bool = False,
        subtract_black: bool = True,
        apply_awb: bool = True,
        awb_gains_override: AwbGains | None = None,
        apply_ccm: bool = True,
        kernel: DemosaicKernel = DemosaicKernel.BILINEAR,
        gamma: bool | float = True,
        exposure: float = 0.0,
        apply_orientation: bool = True,
        denoise: DenoiseKernel | None = None,
        denoise_sigma: float = 0.05,
        denoise_tile_size: int = 512,
        on_step: Callable[[str], None] | None = None,
        on_advance: Callable[[int], None] | None = None,
    ) -> None:
        """Save as PNG.

        raw=True  → 16-bit grayscale Bayer PNG (no demosaic; full bit depth)
        raw=False → 8-bit RGB PNG (debayered, AWB-corrected, CCM-corrected, gamma-encoded)

        apply_ccm: True (default) applies the factory forward_matrix (sensor RGB → XYZ D50 →
                   linear sRGB) when a color profile is available.
        kernel: demosaicing algorithm. BILINEAR (default) needs no extra deps; MALVAR/MENON/DDFAPD
                require ``pip install shadow[demosaic]``. Ignored with raw=True or half_res=True.
        gamma: True  = sRGB transfer function (default)
               False = linear (will look very dark)
               float = simple power-law, e.g. gamma=2.2 → v^(1/2.2)
        exposure: EV stop adjustment applied before gamma (+1.0 = twice as bright)
        awb_gains_override: if provided, use these gains instead of the file's gains
        apply_orientation: True (default) honours flip_h/flip_v from the sensor proto.
        denoise: optional denoising algorithm applied in linear light before CCM.
                 Requires ``pip install shadow[denoise]``.
        denoise_sigma: noise sigma for the denoiser (0.02 subtle – 0.15 heavy).
        denoise_tile_size: spatial tile size for DnCNN/DRUNet (default 512). Increase to
                           1024/2048 on high-VRAM GPUs; reduce to 256 if you run out of VRAM.
        on_step: optional callback invoked with a stage label at each pipeline step.
        on_advance: optional callback invoked with the number of ops completed; called once
                    per tile for DnCNN/DRUNet or once per stage for all other kernels.

        Note: Pillow does not support 16-bit RGB PNG natively. Use to_tiff()
        for 16-bit per-channel debayered output.
        """
        _step = on_step if on_step is not None else lambda _: None
        _adv = on_advance if on_advance is not None else lambda n: None
        path = str(path)
        if raw:
            _step("saving")
            white = self.white_level if subtract_black else _10BIT_MAX
            arr = self.to_raw_numpy(subtract_black=subtract_black)
            scaled = (arr.astype(np.float32) * (65535.0 / white)).clip(0, 65535).astype(np.uint16)
            PILImage.fromarray(scaled).save(path)
            _adv(1)
        else:
            rgb8 = self._export_rgb8(
                half_res=half_res, subtract_black=subtract_black,
                apply_awb=apply_awb, awb_gains_override=awb_gains_override,
                apply_ccm=apply_ccm, kernel=kernel, gamma=gamma, exposure=exposure,
                apply_orientation=apply_orientation,
                denoise=denoise, denoise_sigma=denoise_sigma,
                denoise_tile_size=denoise_tile_size,
                on_step=on_step, on_advance=on_advance,
            )
            _step("saving")
            PILImage.fromarray(rgb8, mode="RGB").save(path)
            _adv(1)

    def to_tiff(
        self,
        path: str | Path,
        *,
        raw: bool = False,
        half_res: bool = False,
        subtract_black: bool = True,
        apply_awb: bool = True,
        awb_gains_override: AwbGains | None = None,
        apply_ccm: bool = True,
        kernel: DemosaicKernel = DemosaicKernel.BILINEAR,
        gamma: bool | float = True,
        exposure: float = 0.0,
        apply_orientation: bool = True,
        denoise: DenoiseKernel | None = None,
        denoise_sigma: float = 0.05,
        denoise_tile_size: int = 512,
        on_step: Callable[[str], None] | None = None,
        on_advance: Callable[[int], None] | None = None,
    ) -> None:
        """Save as TIFF.

        raw=True  → 16-bit grayscale Bayer TIFF (full bit depth, scaled to uint16)
        raw=False → 8-bit RGB TIFF (debayered, AWB-corrected, CCM-corrected, gamma-encoded)

        Same apply_ccm / kernel / gamma / exposure / awb_gains_override / apply_orientation /
        denoise / denoise_sigma / denoise_tile_size / on_step / on_advance semantics as to_png().

        For 16-bit per-channel RGB TIFF, use to_raw_numpy() with the
        `tifffile` library directly.
        """
        _step = on_step if on_step is not None else lambda _: None
        _adv = on_advance if on_advance is not None else lambda n: None
        path = str(path)
        if raw:
            _step("saving")
            white = self.white_level if subtract_black else _10BIT_MAX
            arr = self.to_raw_numpy(subtract_black=subtract_black)
            scaled = (arr.astype(np.float32) * (65535.0 / white)).clip(0, 65535).astype(np.uint16)
            PILImage.fromarray(scaled).save(path)
            _adv(1)
        else:
            rgb8 = self._export_rgb8(
                half_res=half_res, subtract_black=subtract_black,
                apply_awb=apply_awb, awb_gains_override=awb_gains_override,
                apply_ccm=apply_ccm, kernel=kernel, gamma=gamma, exposure=exposure,
                apply_orientation=apply_orientation,
                denoise=denoise, denoise_sigma=denoise_sigma,
                denoise_tile_size=denoise_tile_size,
                on_step=on_step, on_advance=on_advance,
            )
            _step("saving")
            PILImage.fromarray(rgb8, mode="RGB").save(path)
            _adv(1)


