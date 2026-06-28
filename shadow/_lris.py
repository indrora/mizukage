"""LrisFile — reader for the Light L16 LRIS sidecar format.

LRIS files are written by the Lumen desktop application and contain a
depth map (quantized disparity) and refined per-capture calibration data.
They are NOT written by the camera itself.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

LRIS_MAGIC = 0x12345678
# Depth map starts at offset 0x28 (40 bytes into the file)
DEPTH_OFFSET = 0x28
DEPTH_COLS = 260
DEPTH_ROWS = 195


class LrisFile:
    """Reader for a Light L16 LRIS sidecar file.

    Typical usage::

        lris = LrisFile.open("photo.lris")
        dm = lris.depth_map        # int32 (195, 260) array
        valid = dm[dm >= 0]        # positive values = valid disparity
    """

    def __init__(self, depth_map: np.ndarray) -> None:
        self._depth_map = depth_map

    @classmethod
    def open(cls, path: str | Path) -> "LrisFile":
        """Open an LRIS sidecar file."""
        data = Path(path).read_bytes()
        return cls._parse(data)

    @classmethod
    def _parse(cls, data: bytes) -> "LrisFile":
        if len(data) < DEPTH_OFFSET:
            raise ValueError(f"LRIS file too short ({len(data)} bytes)")

        magic = struct.unpack_from("<I", data, 0)[0]
        if magic != LRIS_MAGIC:
            raise ValueError(
                f"Not an LRIS file: magic {magic:#010x}, expected {LRIS_MAGIC:#010x}"
            )

        n_values = DEPTH_COLS * DEPTH_ROWS
        n_bytes = n_values * 4  # int32 = 4 bytes each
        end = DEPTH_OFFSET + n_bytes

        if len(data) < end:
            raise ValueError(
                f"LRIS file too short for depth map: need {end} bytes, got {len(data)}"
            )

        depth = np.frombuffer(data[DEPTH_OFFSET:end], dtype="<i4").reshape(
            DEPTH_ROWS, DEPTH_COLS
        ).copy()
        return cls(depth)

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def depth_map(self) -> np.ndarray:
        """Quantized disparity map as int32 array of shape (195, 260).

        Negative values indicate invalid/occluded pixels. Positive values
        are quantized disparity from Lumen's stereo reconstruction; convert
        to metric depth via the camera's baseline and focal length.
        """
        return self._depth_map

    @property
    def depth_shape(self) -> tuple[int, int]:
        """(rows, cols) of the depth map — always (195, 260)."""
        return (DEPTH_ROWS, DEPTH_COLS)

    @property
    def valid_fraction(self) -> float:
        """Fraction of depth map pixels with non-negative (valid) disparity."""
        return float(np.sum(self._depth_map >= 0)) / self._depth_map.size

    @property
    def disparity_range(self) -> tuple[int, int]:
        """(min, max) disparity among valid (non-negative) pixels.

        Returns (0, 0) if no valid pixels exist.
        """
        valid = self._depth_map[self._depth_map >= 0]
        if valid.size == 0:
            return (0, 0)
        return (int(valid.min()), int(valid.max()))
