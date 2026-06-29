"""Bayer demosaicing — numpy-native algorithms plus optional colour-demosaicing backends.

Built-in functions (debayer_half, debayer_bilinear) have no external dependencies
beyond numpy and are always available.

Higher-quality algorithms (Malvar2004, Menon2007, DDFAPD) are available when
the ``colour-demosaicing`` package is installed (``pip install shadow[demosaic]``).
Pass a DemosaicKernel to to_debayered_numpy() or the export functions to select one.
"""
from __future__ import annotations

from enum import Enum

import numpy as np


class DemosaicKernel(str, Enum):
    """Demosaicing algorithm selector.

    Built-in (no extra deps):
      HALF       — fast half-resolution subsampling (good for previews)
      BILINEAR   — full-resolution bilinear (default)

    Requires colour-demosaicing (``pip install shadow[demosaic]``):
      MALVAR     — Malvar-He-Cutler 2004 (fast, high quality; recommended upgrade)
      MENON      — Menon et al. 2007 (slower, fewer colour fringing artefacts)
      DDFAPD     — Adaptive Directed interpolation (highest quality, slowest)
    """

    HALF = "half"
    BILINEAR = "bilinear"
    MALVAR = "malvar"
    MENON = "menon"
    DDFAPD = "ddfapd"


# Maps (r_row, r_col) to the CFA pattern strings used by colour-demosaicing.
_CFA_STR: dict[tuple[int, int], str] = {
    (0, 0): "RGGB",
    (0, 1): "GRBG",
    (1, 0): "GBRG",
    (1, 1): "BGGR",
}


def debayer_half(bayer: np.ndarray, r_row: int, r_col: int) -> np.ndarray:
    """Fast half-resolution demosaic by subsampling each Bayer channel.

    Returns float32 array of shape (H/2, W/2, 3) — R, G, B.
    Suitable for quick previews; output is half the sensor resolution.
    """
    b_row, b_col = 1 - r_row, 1 - r_col
    f = bayer.astype(np.float32)

    R = f[r_row::2, r_col::2]
    B = f[b_row::2, b_col::2]
    # Average the two green sub-channels
    G = (f[r_row::2, b_col::2] + f[b_row::2, r_col::2]) * 0.5

    return np.stack([R, G, B], axis=2)


def debayer_bilinear(bayer: np.ndarray, r_row: int, r_col: int) -> np.ndarray:
    """Full-resolution bilinear demosaic.

    Returns float32 array of shape (H, W, 3) with values in the same range as
    the input (e.g. [0..959] after black-level subtraction from 10-bit data).
    """
    H, W = bayer.shape
    f = bayer.astype(np.float32)
    b_row, b_col = 1 - r_row, 1 - r_col

    # ── Red ──────────────────────────────────────────────────────────────────
    R = np.zeros((H, W), np.float32)
    R[r_row::2, r_col::2] = f[r_row::2, r_col::2]  # known R

    # Horizontal neighbours at G1 positions (same row as R, adjacent col)
    R[r_row::2, b_col::2] = 0.5 * (
        _roll_col(R[r_row::2, :], +1)[:, b_col::2]
        + _roll_col(R[r_row::2, :], -1)[:, b_col::2]
    )
    # Vertical neighbours at G2 positions (same col as R, adjacent row)
    R[b_row::2, r_col::2] = 0.5 * (
        _roll_row(R[:, r_col::2], +1)[b_row::2, :]
        + _roll_row(R[:, r_col::2], -1)[b_row::2, :]
    )
    # Bilinear at B positions (average of 4 diagonal R neighbours)
    R[b_row::2, b_col::2] = 0.25 * (
        R[r_row::2, r_col::2]
        + _roll_col(R[r_row::2, r_col::2], -1)
        + _roll_row(R[r_row::2, r_col::2], -1)
        + _roll_col(_roll_row(R[r_row::2, r_col::2], -1), -1)
    )

    # ── Blue ──────────────────────────────────────────────────────────────────
    B = np.zeros((H, W), np.float32)
    B[b_row::2, b_col::2] = f[b_row::2, b_col::2]  # known B

    B[b_row::2, r_col::2] = 0.5 * (
        _roll_col(B[b_row::2, :], +1)[:, r_col::2]
        + _roll_col(B[b_row::2, :], -1)[:, r_col::2]
    )
    B[r_row::2, b_col::2] = 0.5 * (
        _roll_row(B[:, b_col::2], +1)[r_row::2, :]
        + _roll_row(B[:, b_col::2], -1)[r_row::2, :]
    )
    B[r_row::2, r_col::2] = 0.25 * (
        B[b_row::2, b_col::2]
        + _roll_col(B[b_row::2, b_col::2], -1)
        + _roll_row(B[b_row::2, b_col::2], -1)
        + _roll_col(_roll_row(B[b_row::2, b_col::2], -1), -1)
    )

    # ── Green ─────────────────────────────────────────────────────────────────
    G = np.zeros((H, W), np.float32)
    G[r_row::2, b_col::2] = f[r_row::2, b_col::2]  # G1 (known)
    G[b_row::2, r_col::2] = f[b_row::2, r_col::2]  # G2 (known)

    # At R and B positions: average of 4 cross-shaped (NSEW) neighbours in G
    for pr, pc in [(r_row, r_col), (b_row, b_col)]:
        G[pr::2, pc::2] = 0.25 * (
            np.roll(G, -1, axis=0)[pr::2, pc::2]
            + np.roll(G, +1, axis=0)[pr::2, pc::2]
            + np.roll(G, -1, axis=1)[pr::2, pc::2]
            + np.roll(G, +1, axis=1)[pr::2, pc::2]
        )

    return np.stack([R, G, B], axis=2)


def debayer_colour(
    bayer: np.ndarray,
    r_row: int,
    r_col: int,
    kernel: DemosaicKernel = DemosaicKernel.MALVAR,
) -> np.ndarray:
    """Demosaic using a colour-demosaicing algorithm.

    Requires ``pip install shadow[demosaic]`` (colour-demosaicing package).

    Returns float32 (H, W, 3) with the same value range as the input.
    """
    try:
        import colour_demosaicing as cd
    except ImportError as exc:
        raise ImportError(
            f"kernel={kernel.value!r} requires colour-demosaicing: "
            "pip install 'shadow[demosaic]'"
        ) from exc

    pattern = _CFA_STR.get((r_row, r_col))
    if pattern is None:
        raise ValueError(f"Unknown Bayer pattern: r_row={r_row}, r_col={r_col}")

    # colour-demosaicing works in float64; input should already be float32.
    cfa = bayer.astype(np.float64)

    match kernel:
        case DemosaicKernel.MALVAR:
            rgb = cd.demosaicing_CFA_Bayer_Malvar2004(cfa, pattern)
        case DemosaicKernel.MENON:
            rgb = cd.demosaicing_CFA_Bayer_Menon2007(cfa, pattern)
        case DemosaicKernel.DDFAPD:
            rgb = cd.demosaicing_CFA_Bayer_DDFAPD(cfa, pattern)
        case _:  # pragma: no cover
            raise ValueError(f"Not a colour-demosaicing kernel: {kernel!r}")

    return rgb.astype(np.float32)


def _roll_row(arr: np.ndarray, shift: int) -> np.ndarray:
    """Roll along axis=0 (rows) with edge replication at boundaries."""
    if shift == 0:
        return arr
    if shift > 0:
        return np.concatenate([arr[:1, :]] * shift + [arr[:-shift, :]], axis=0)
    return np.concatenate([arr[-shift:, :], arr[-1:, :] * (-shift)], axis=0)


def _roll_col(arr: np.ndarray, shift: int) -> np.ndarray:
    """Roll along axis=1 (columns) with edge replication at boundaries."""
    if shift == 0:
        return arr
    if shift > 0:
        return np.concatenate([arr[:, :1]] * shift + [arr[:, :-shift]], axis=1)
    return np.concatenate([arr[:, -shift:], arr[:, -1:] * (-shift)], axis=1)
