"""
Binary serialisation of Python objects (dict, list, primitives) using
varint encoding and type tags. Used for storing values in segments.
"""

import struct
from typing import Any, Tuple

from .constants import (
    TYPE_NULL, TYPE_BOOL, TYPE_INT, TYPE_FLOAT,
    TYPE_STRING, TYPE_BYTES, TYPE_LIST, TYPE_DICT
)

# ---------- Varint helpers ----------
def _write_varint(value: int) -> bytes:
    value = int(value)
    result = bytearray()
    while value >= 0x80:
        result.append((value & 0x7F) | 0x80)
        value >>= 7
    result.append(value & 0x7F)
    return bytes(result)

def _read_varint(data: bytes, pos: int) -> Tuple[int, int]:
    value = 0
    shift = 0
    while True:
        if pos >= len(data):
            raise ValueError("Truncated varint")
        b = data[pos]
        pos += 1
        value |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
    return value, pos

def _write_signed_varint(value: int) -> bytes:
    value = int(value)
    return _write_varint((value << 1) ^ (value >> 63))

def _read_signed_varint(data: bytes, pos: int) -> Tuple[int, int]:
    value, pos = _read_varint(data, pos)
    return ((value >> 1) ^ -(value & 1)), pos

# ---------- Public encoder / decoder ----------
def encode_object(obj: Any) -> bytes:
    """Encode any Python object into a binary blob."""
    if obj is None:
        return bytes([TYPE_NULL])
    if isinstance(obj, bool):
        return bytes([TYPE_BOOL, 1 if obj else 0])
    if isinstance(obj, int):
        return bytes([TYPE_INT]) + _write_signed_varint(obj)
    if isinstance(obj, float):
        return bytes([TYPE_FLOAT]) + struct.pack("<d", obj)
    if isinstance(obj, str):
        b = obj.encode('utf-8')
        return bytes([TYPE_STRING]) + _write_varint(len(b)) + b
    if isinstance(obj, bytes):
        return bytes([TYPE_BYTES]) + _write_varint(len(obj)) + obj
    if isinstance(obj, list):
        parts = [bytes([TYPE_LIST]), _write_varint(len(obj))]
        for item in obj:
            parts.append(encode_object(item))
        return b''.join(parts)
    if isinstance(obj, dict):
        parts = [bytes([TYPE_DICT]), _write_varint(len(obj))]
        for key, value in obj.items():
            parts.append(encode_object(key))
            parts.append(encode_object(value))
        return b''.join(parts)
    raise TypeError(f"Unsupported type for binary serialization: {type(obj)}")

def decode_object(data: bytes) -> Any:
    """Decode a binary blob back into a Python object."""
    pos = 0
    obj, pos = _decode_object_at(data, pos)
    if pos != len(data):
        raise ValueError("Trailing data detected after deserialization")
    return obj

def _decode_object_at(data: bytes, pos: int) -> Tuple[Any, int]:
    if pos >= len(data):
        raise ValueError("Unexpected end of binary payload")
    type_code = data[pos]
    pos += 1

    if type_code == TYPE_NULL:
        return None, pos
    if type_code == TYPE_BOOL:
        if pos >= len(data):
            raise ValueError("Truncated boolean value")
        return bool(data[pos]), pos + 1
    if type_code == TYPE_INT:
        return _read_signed_varint(data, pos)
    if type_code == TYPE_FLOAT:
        if pos + 8 > len(data):
            raise ValueError("Truncated float value")
        return struct.unpack("<d", data[pos:pos+8])[0], pos + 8
    if type_code == TYPE_STRING:
        length, pos = _read_varint(data, pos)
        if pos + length > len(data):
            raise ValueError("Truncated string value")
        s = data[pos:pos+length].decode('utf-8')
        return s, pos + length
    if type_code == TYPE_BYTES:
        length, pos = _read_varint(data, pos)
        if pos + length > len(data):
            raise ValueError("Truncated byte array value")
        return data[pos:pos+length], pos + length
    if type_code == TYPE_LIST:
        length, pos = _read_varint(data, pos)
        result = []
        for _ in range(length):
            item, pos = _decode_object_at(data, pos)
            result.append(item)
        return result, pos
    if type_code == TYPE_DICT:
        length, pos = _read_varint(data, pos)
        result = {}
        for _ in range(length):
            key, pos = _decode_object_at(data, pos)
            value, pos = _decode_object_at(data, pos)
            result[key] = value
        return result, pos
    raise ValueError(f"Unknown serialization type code encountered: {type_code}")