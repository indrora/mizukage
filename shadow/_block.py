"""LELR block header parser.

Single responsibility: parse the 32-byte binary LELR block structure.
No protobuf, no image data.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Iterator

LELR_MAGIC = b"LELR"
HEADER_SIZE = 32

# Struct layout (32 bytes total):
#   4s = magic "LELR"
#   Q  = block_length (u64 LE) — total bytes including this header
#   Q  = msg_offset (u64 LE)   — from block start to protobuf payload
#   I  = msg_len (u32 LE)      — protobuf payload length in bytes
#   B  = msg_type (u8)
#   7x = 7 padding bytes
_HEADER_STRUCT = struct.Struct("<4sQQIB7x")
assert _HEADER_STRUCT.size == HEADER_SIZE


class BlockType(IntEnum):
    LIGHT_HEADER = 0
    VIEW_PREFERENCES = 1
    GPS_DATA = 2


@dataclass(slots=True)
class BlockHeader:
    block_length: int  # total bytes in block including this 32-byte header
    msg_offset: int    # offset from block start (not file start) to protobuf bytes
    msg_len: int       # protobuf payload length
    msg_type: BlockType


def parse_block_header(data: bytes | memoryview, offset: int = 0) -> BlockHeader:
    magic, block_len, msg_off, msg_len, msg_type_byte = _HEADER_STRUCT.unpack_from(data, offset)
    if magic != LELR_MAGIC:
        raise ValueError(f"Expected LELR magic at {offset:#010x}, got {bytes(magic)!r}")
    try:
        msg_type = BlockType(msg_type_byte)
    except ValueError:
        raise ValueError(f"Unknown block type {msg_type_byte} at {offset:#010x}")
    return BlockHeader(
        block_length=int(block_len),
        msg_offset=int(msg_off),
        msg_len=int(msg_len),
        msg_type=msg_type,
    )


def iter_blocks(data: bytes) -> Iterator[tuple[int, BlockHeader]]:
    """Yield (block_start_offset, header) for each consecutive LELR block.

    Stops at the first non-LELR offset or end of data.
    """
    pos = 0
    while pos + HEADER_SIZE <= len(data):
        if data[pos : pos + 4] != LELR_MAGIC:
            break
        hdr = parse_block_header(data, pos)
        if hdr.block_length == 0:
            break
        yield pos, hdr
        pos += hdr.block_length
