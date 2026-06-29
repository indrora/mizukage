"""shadow — Light L16 LRI/LRIS camera file reader.

Quick start::

    import shadow

    lri = shadow.open_lri("photo.lri")
    print(lri.metadata.focal_length_mm)

    ref = lri.reference_image
    ref.to_png("reference.png")

    lris = shadow.open_lris("photo.lris")
    print(lris.depth_map.shape)
"""
from shadow._debayer import DemosaicKernel
from shadow._denoise import DenoiseKernel, DenoiseFn, make_tiled_denoiser
from shadow._lri import LriFile
from shadow._lris import LrisFile
from shadow._image import RawImage
from shadow._types import (
    AwbGains,
    AwbMode,
    BayerPattern,
    CameraId,
    CaptureMetadata,
    ColorProfile,
    GpsData,
    HdrMode,
    Illuminant,
    ModuleCapture,
    Orientation,
    RawFormat,
    SceneMode,
    SensorModel,
)

__all__ = [
    # File readers
    "LriFile",
    "LrisFile",
    "open_lri",
    "open_lris",
    # Image class
    "RawImage",
    # Demosaicing
    "DemosaicKernel",
    # Denoising
    "DenoiseKernel",
    "DenoiseFn",
    "make_tiled_denoiser",
    # Enums and data types
    "CameraId",
    "SensorModel",
    "RawFormat",
    "BayerPattern",
    "Illuminant",
    "HdrMode",
    "SceneMode",
    "AwbMode",
    "Orientation",
    # Dataclasses
    "AwbGains",
    "ColorProfile",
    "GpsData",
    "CaptureMetadata",
    "ModuleCapture",
]


def open_lri(path: str) -> LriFile:
    """Open an LRI file and return an LriFile reader."""
    return LriFile.open(path)


def open_lris(path: str) -> LrisFile:
    """Open an LRIS sidecar file and return an LrisFile reader."""
    return LrisFile.open(path)
