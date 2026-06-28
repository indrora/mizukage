"""Bayer demosaicing — numpy only, no OpenCV dependency.

Both functions accept a raw uint16 Bayer array (height, width) and the
R pixel position (r_row, r_col) within the 2×2 CFA tile.
"""
from __future__ import annotations

import numpy as np


def debayer_half(bayer: np.ndarray, r_row: int, r_col: int) -> np.ndarray:
    """Fast half-resolution demosaic by subsampling each Bayer channel.

    Returns uint16 array of shape (H/2, W/2, 3) — R, G, B.
    Suitable for quick previews; output is half the sensor resolution.
    """
    b_row, b_col = 1 - r_row, 1 - r_col

    R = bayer[r_row::2, r_col::2].astype(np.uint16)
    B = bayer[b_row::2, b_col::2].astype(np.uint16)
    # Average the two green sub-channels
    G = (
        (bayer[r_row::2, b_col::2].astype(np.uint32) + bayer[b_row::2, r_col::2])
        >> 1
    ).astype(np.uint16)

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
