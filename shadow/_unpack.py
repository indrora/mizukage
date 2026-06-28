"""Raw pixel data unpacking.

Both functions return np.ndarray with dtype=uint16, shape=(height, width).
"""
from __future__ import annotations

import io
import struct

import numpy as np
from PIL import Image as PILImage

BJPG_MAGIC = b"BJPG"
# Header layout:
#   [0:4]   "BJPG" magic
#   [4:8]   format u32 LE (0=colour/4-JPEG, 1=mono/1-JPEG)
#   [8:24]  four u32 LE JPEG lengths
#   [24:1576] 1552 bytes of unknown/padding
BJPG_HEADER_SIZE = 1576


def unpack_10bpp(
    data: bytes,
    abs_offset: int,
    width: int,
    height: int,
    stride: int,
) -> np.ndarray:
    """Vectorised 10-bit packed Bayer → uint16 (height, width).

    Layout: every 4 consecutive pixels occupy 5 bytes (little-endian bit order):
        pixel0 = b0        | ((b1 & 0x03) << 8)
        pixel1 = (b1 >> 2) | ((b2 & 0x0F) << 6)
        pixel2 = (b2 >> 4) | ((b3 & 0x3F) << 4)
        pixel3 = (b3 >> 6) |  (b4          << 2)
    """
    raw = np.frombuffer(
        data, dtype=np.uint8, offset=abs_offset, count=stride * height
    ).reshape(height, stride)

    groups_per_row = width // 4
    g = raw[:, : groups_per_row * 5].reshape(height, groups_per_row, 5).astype(np.uint16)

    p0 = g[:, :, 0]        | ((g[:, :, 1] & 0x03) << 8)
    p1 = (g[:, :, 1] >> 2) | ((g[:, :, 2] & 0x0F) << 6)
    p2 = (g[:, :, 2] >> 4) | ((g[:, :, 3] & 0x3F) << 4)
    p3 = (g[:, :, 3] >> 6) |  (g[:, :, 4]         << 2)

    return np.stack([p0, p1, p2, p3], axis=2).reshape(height, width)


def decode_bjpg(
    data: bytes,
    abs_offset: int,
    width: int,
    height: int,
    r_row: int,
    r_col: int,
) -> np.ndarray:
    """Decode a BayerJPEG block → uint16 Bayer (height, width).

    For colour sensors: 4 half-resolution JPEGs (one per Bayer channel) are
    decoded and interleaved into a full-resolution Bayer grid.
    For mono sensors (colour_fmt==1): a single full-resolution JPEG.

    r_row, r_col: position of the R pixel in the 2×2 Bayer tile (0 or 1 each).
    """
    hdr = data[abs_offset : abs_offset + BJPG_HEADER_SIZE]
    if hdr[:4] != BJPG_MAGIC:
        raise ValueError(
            f"Expected BJPG magic at {abs_offset:#010x}, got {hdr[:4]!r}"
        )

    colour_fmt = struct.unpack_from("<I", hdr, 4)[0]   # 0=colour, 1=mono
    jpeg_lens = struct.unpack_from("<4I", hdr, 8)       # up to 4 JPEG lengths

    pos = abs_offset + BJPG_HEADER_SIZE
    n_jpegs = 1 if colour_fmt == 1 else 4

    channels: list[np.ndarray] = []
    for i in range(n_jpegs):
        length = jpeg_lens[i]
        if length == 0:
            break
        img = PILImage.open(io.BytesIO(data[pos : pos + length])).convert("L")
        # JPEGs are 8-bit; scale to 10-bit range [0..1020] to match PACKED_10BPP
        arr = (np.array(img, dtype=np.uint32) * 4).clip(0, 1023).astype(np.uint16)
        channels.append(arr)
        pos += length

    if colour_fmt == 1 or len(channels) == 1:
        # Mono: single full-resolution JPEG, return as-is
        return channels[0]

    # Colour: 4 half-res channels (H/2, W/2) — interleave into full Bayer (H, W)
    # JPEG order from reference: [R, G1, G2, B]
    # r_row/r_col positions the R pixel; B is diagonally opposite.
    H2, W2 = channels[0].shape
    out = np.zeros((H2 * 2, W2 * 2), dtype=np.uint16)
    b_row, b_col = 1 - r_row, 1 - r_col

    out[r_row::2, r_col::2] = channels[0]  # R
    out[r_row::2, b_col::2] = channels[1]  # G1  (same row as R, opposite col)
    out[b_row::2, r_col::2] = channels[2]  # G2  (same col as R, opposite row)
    out[b_row::2, b_col::2] = channels[3]  # B
    return out
